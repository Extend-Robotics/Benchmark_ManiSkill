import os
import boto3
from tqdm import tqdm
from huggingface_hub import (
    list_datasets,
    hf_hub_download,
    hf_hub_url,
    login,
    list_repo_files,
    hf_api,
)
import concurrent.futures
from botocore.exceptions import ClientError
from mypy_boto3_s3.client import S3Client


class LeRobotFormatDatasetUploader:
    def __init__(
        self,
        s3_bucket: str,
        hf_user: str = "lerobot",
        local_cache_dir: str = "~/.cache/huggingface/lerobot/",
        max_datasets: int | None = None,
        hf_token: str | None = None,
        upload_local_only: bool = False,
        aws_profile: str | None = None,
    ):
        """
        Initializes the uploader.

        :param s3_bucket: Name of the target S3 bucket.
        :param hf_user: Hugging Face username (default: 'lerobot').
        :param local_cache_dir: Path to local datasets (if any exist).
        :param max_datasets: Number of datasets to process (for testing).
        :param hf_token: Hugging Face authentication token for logging in.
        :param upload_local_only: Flag to upload only local datasets.
        :param aws_profile: Optional AWS profile name (if using multiple AWS accounts).
        """
        self.s3_bucket = s3_bucket
        self.hf_user = hf_user
        self.local_cache_dir = os.path.expanduser(local_cache_dir)
        self.s3_client = self.create_s3_client(aws_profile=aws_profile)
        self.max_datasets = max_datasets
        self.hf_token = hf_token
        self.upload_local_only = upload_local_only

        if self.hf_token:
            login(self.hf_token)

    def create_s3_client(self, aws_profile: str | None = None) -> S3Client:
        """Creates an S3 client using environment variables or AWS profile."""
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            s3_client = session.client("s3")
        else:
            # Using default profile
            s3_client = boto3.client("s3")
        return s3_client

    def check_s3_bucket_access(self):
        """Checks if the user has access to the S3 bucket."""
        try:
            # Attempt to list objects in the bucket to verify access
            self.s3_client.list_objects_v2(Bucket=self.s3_bucket)
            print(f"✔ Access to S3 bucket {self.s3_bucket} confirmed.")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "AccessDenied":
                raise PermissionError(
                    f"❌ Access denied to S3 bucket {self.s3_bucket}. Please check your AWS credentials."
                )
            elif error_code == "NoSuchBucket":
                raise FileNotFoundError(
                    f"❌ S3 bucket {self.s3_bucket} does not exist."
                )
            else:
                raise Exception(f"❌ Error accessing S3 bucket {self.s3_bucket}: {e}")

    def list_huggingface_datasets(self) -> list[hf_api.DatasetInfo]:
        """Fetches dataset names from Hugging Face."""
        datasets = list(list_datasets(author=self.hf_user))
        return datasets[: self.max_datasets] if self.max_datasets else datasets

    def file_exists_on_s3(self, s3_key: str) -> bool:
        """Checks if the file already exists in the target S3 bucket."""
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    def upload_file_to_s3(self, file_path: str, s3_key: str):
        """Uploads a given file to S3 only if it doesn't already exist."""
        if self.file_exists_on_s3(s3_key):
            print(f"⚠ {s3_key} already exists on S3, skipping upload...")
            return

        try:
            print(f"Uploading {file_path} to s3://{self.s3_bucket}/{s3_key}...")
            with open(file_path, "rb") as data:
                self.s3_client.upload_fileobj(data, self.s3_bucket, s3_key)
            print(f"✔ Uploaded {file_path}")
        except Exception as e:
            print(f"❌ Failed to upload {file_path}: {e}")

    def process_huggingface_datasets(self) -> None:
        """Streams datasets from Hugging Face and uploads to S3."""
        datasets = self.list_huggingface_datasets()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for dataset in tqdm(datasets, desc="Processing Hugging Face datasets"):
                dataset_id = dataset.id
                futures.append(executor.submit(self.process_dataset_files, dataset_id))

            for future in concurrent.futures.as_completed(futures):
                future.result()

    def process_dataset_files(self, dataset_id: str) -> None:
        """Processes the files in a dataset from Hugging Face and uploads to S3."""
        print(f"\n🔄 Processing: {dataset_id}")

        try:
            # List available files
            file_paths = list_repo_files(repo_id=dataset_id, repo_type="dataset")

            if not file_paths:
                print(f"⚠ No files found in {dataset_id}, skipping...")
                return

            for file_path in tqdm(
                file_paths, desc=f"Streaming {dataset_id}", leave=False
            ):
                # Construct correct file URL
                file_url = hf_hub_url(
                    repo_id=dataset_id, filename=file_path, repo_type="dataset"
                )
                # print(f"🔗 Downloading: {file_url}")  # Debugging: Print URL

                # Download file
                file_stream = hf_hub_download(
                    repo_id=dataset_id, filename=file_path, repo_type="dataset"
                )

                # Upload to S3
                s3_key = f"public/{dataset_id}/{file_path}"
                self.upload_file_to_s3(file_stream, s3_key)

        except Exception as e:
            print(f"❌ Error processing {dataset_id}: {e}")

    def process_local_datasets(self) -> None:
        """Uploads datasets found in the local cache directory using parallel threads."""
        if not os.path.exists(self.local_cache_dir):
            print(f"⚠ Local dataset directory {self.local_cache_dir} does not exist.")
            return

        print(f"\n🔄 Processing local datasets from {self.local_cache_dir}...")

        local_files = []
        for root, _, files in os.walk(self.local_cache_dir):
            for file in files:
                local_files.append(os.path.join(root, file))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for local_path in tqdm(local_files, desc="Uploading local datasets"):
                s3_key = f"private/{os.path.relpath(local_path, self.local_cache_dir)}"
                futures.append(
                    executor.submit(self.upload_file_to_s3, local_path, s3_key)
                )

            for future in concurrent.futures.as_completed(futures):
                future.result()

    def run(self) -> None:
        """Runs both Hugging Face and local dataset uploads based on the flag."""
        self.check_s3_bucket_access()

        if self.upload_local_only:
            print("🔄 Uploading only local datasets...\n")
            self.process_local_datasets()
        else:
            print("🔄 Uploading Hugging Face and local datasets...\n")
            self.process_huggingface_datasets()
            self.process_local_datasets()

        print("\n✅ All datasets uploaded successfully!")


def main():
    uploader = LeRobotFormatDatasetUploader(
        s3_bucket="er-robotics-dataset-hub",
        upload_local_only=False,
        aws_profile="default",
    )

    uploader.run()


if __name__ == "__main__":
    main()

# src/gcp_clients/storage_client.py
from google.cloud import storage
import os

class GCSClient:
    def __init__(self, project_id: str, service_account_key_path: str):
        # Service account key path is relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        credentials_full_path = os.path.join(project_root, service_account_key_path)

        if not os.path.exists(credentials_full_path):
            raise FileNotFoundError(f"GCSClient: Service account key not found at {credentials_full_path}")
        
        # Explicitly set environment variable for this client instance context if needed,
        # though storage.Client() can also take credentials directly.
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_full_path
        self.client = storage.Client.from_service_account_json(credentials_full_path, project=project_id)
        self.project_id = project_id
        print(f"GCSClient initialized for project: {self.project_id}")

    def check_or_create_bucket(self, bucket_name: str, location: str = "US"):
        try:
            bucket = self.client.bucket(bucket_name)
            if bucket.exists():
                print(f"GCS Bucket '{bucket_name}' already exists.")
                return bucket
            else:
                print(f"GCS Bucket '{bucket_name}' does not exist. Creating in location '{location}'...")
                new_bucket = self.client.create_bucket(bucket_name, location=location)
                print(f"Bucket '{new_bucket.name}' created successfully.")
                return new_bucket
        except Exception as e:
            print(f"Error checking or creating GCS bucket '{bucket_name}': {e}")
            raise

    def upload_file(self, bucket_name: str, source_file_path: str, destination_blob_name: str):
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_path)
            print(f"File {source_file_path} uploaded to {bucket_name}/{destination_blob_name}.")
        except Exception as e:
            print(f"Error uploading file to GCS: {e}")
            raise

    def download_text_file(self, bucket_name: str, source_blob_name: str) -> str | None:
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            if not blob.exists():
                print(f"Error: File '{source_blob_name}' does not exist in bucket '{bucket_name}'.")
                return None
            file_content_bytes = blob.download_as_bytes()
            # Attempt to decode as UTF-8, can add more robust decoding later
            return file_content_bytes.decode('utf-8')
        except Exception as e:
            print(f"Error downloading text file '{source_blob_name}' from GCS: {e}")
            return None

    def list_files(self, bucket_name: str, prefix: str = "") -> list[str]:
        try:
            blobs = self.client.list_blobs(bucket_name, prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            print(f"Error listing files in GCS bucket '{bucket_name}': {e}")
            return []
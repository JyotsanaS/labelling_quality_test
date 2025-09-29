from pathlib import Path
import requests
import os
import tempfile
from urllib.parse import urlparse
import distribution.download_constants as constants


class S3Downloader:
    def __init__(self, s3_client):
        self.s3_client = s3_client

    def download_from_s3(self, bucket_name, file_key, download_path, checksum):
        self.s3_client.download_file(bucket_name, file_key, download_path)
        if not checksum:
            pass

    @classmethod
    def download_from_http(cls, remote_path, download_path, checksum):
        if not os.path.exists(download_path):
            print("Downloading model...")
            response = requests.get(remote_path, stream=True)
            response.raise_for_status()
            with open(download_path, "wb") as f:
                f.write(response.content)
            if not checksum:
                pass

    def download_pass(self, path):
        bucket, key = self.get_bucket_key_from_s3_url(path)
        download_path = Path(os.path.join(tempfile.gettempdir(), constants.DIR))
        if not download_path.exists():
            download_path.mkdir(parents=True, exist_ok=True)
        download_path = download_path / key.split("/")[-1]
        self.download_from_s3(bucket, key, download_path, None)
        return download_path

    def get_bucket_key_from_s3_url(self, url):
        parts = urlparse(url)
        bucket = parts.netloc.split(".").pop(0)
        key = parts.path.lstrip("/")
        return bucket, key

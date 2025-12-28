import os
import boto3
from abc import ABC, abstractmethod
from urllib.parse import urlparse

class IOHandler(ABC):
    @abstractmethod
    def read(self, uri: str) -> bytes:
        pass

    @abstractmethod
    def write(self, uri: str, data: bytes) -> None:
        pass

class LocalIOHandler(IOHandler):
    def read(self, uri: str) -> bytes:
        # remove file:// scheme if present
        path = self._parse_path(uri)
        with open(path, 'rb') as f:
            return f.read()

    def write(self, uri: str, data: bytes) -> None:
        path = self._parse_path(uri)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)
    
    def _parse_path(self, uri: str) -> str:
        if uri.startswith('file://'):
            return uri[7:]
        return uri

class S3IOHandler(IOHandler):
    def __init__(self):
        self.s3 = boto3.client('s3')

    def read(self, uri: str) -> bytes:
        bucket, key = self._parse_s3_uri(uri)
        response = self.s3.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()

    def write(self, uri: str, data: bytes) -> None:
        bucket, key = self._parse_s3_uri(uri)
        self.s3.put_object(Bucket=bucket, Key=key, Body=data)

    def _parse_s3_uri(self, uri: str):
        parsed = urlparse(uri)
        return parsed.netloc, parsed.path.lstrip('/')

def get_io_handler(uri: str) -> IOHandler:
    if uri.startswith('s3://'):
        return S3IOHandler()
    return LocalIOHandler()

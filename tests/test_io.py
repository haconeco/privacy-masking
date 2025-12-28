import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
from src.io_handler import LocalIOHandler, S3IOHandler

def test_local_io_read_write():
    handler = LocalIOHandler()
    content = b"fake image"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.jpg")
        
        # Test Write
        handler.write(path, content)
        assert os.path.exists(path)
        
        # Test Read
        read_content = handler.read(path)
        assert read_content == content

def test_s3_io_read():
    # Mock boto3
    with patch('boto3.client') as mock_boto:
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        
        # Mock body.read()
        mock_body = MagicMock()
        mock_body.read.return_value = b"s3 data"
        mock_s3.get_object.return_value = {'Body': mock_body}
        
        handler = S3IOHandler()
        data = handler.read("s3://bucket/test.jpg")
        
        assert data == b"s3 data"
        mock_s3.get_object.assert_called_with(Bucket="bucket", Key="test.jpg")

def test_s3_io_write():
    with patch('boto3.client') as mock_boto:
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        
        handler = S3IOHandler()
        handler.write("s3://bucket/out.jpg", b"output")
        
        mock_s3.put_object.assert_called_with(Bucket="bucket", Key="out.jpg", Body=b"output")

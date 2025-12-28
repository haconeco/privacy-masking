import pytest
from unittest.mock import MagicMock, call
import numpy as np
from src.pipeline import BatchProcessor, ProcessingConfig
from src.manifest import Status
from src.detector import BBox

def test_pipeline_single_process():
    # Mock components
    mock_manifest = MagicMock()
    # 2 pending items
    mock_manifest.get_pending_indices.return_value = [0, 1]
    mock_manifest.get_row.side_effect = [
        {'source_uri': 'in/1.jpg', 'output_uri': 'out/1.jpg'},
        {'source_uri': 'in/2.jpg', 'output_uri': 'out/2.jpg'}
    ]
    
    mock_io = MagicMock()
    mock_io.read.return_value = b'fakeimagebytes'
    
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [BBox(10, 10, 20, 20)]
    
    mock_masker = MagicMock()
    mock_masker.apply_blur.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # We need to mock cv2.imdecode and imencode inside pipeline or mock the calls
    with patch('cv2.imdecode') as mock_decode, \
         patch('cv2.imencode') as mock_encode:
        
        mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        # imencode returns (retval, buffer). buffer is ndarray.
        mock_buffer = MagicMock()
        mock_buffer.tobytes.return_value = b'encodedbytes'
        mock_encode.return_value = (True, mock_buffer)

        config = ProcessingConfig()
        processor = BatchProcessor(config, mock_manifest, mock_io, mock_detector, mock_masker)
        
        processor.run_sequential() # Simplest mode first
        
        # Verify interactions
        assert mock_io.read.call_count == 2
        assert mock_detector.detect.call_count == 2
        assert mock_masker.apply_blur.call_count == 2
        assert mock_io.write.call_count == 2
        
        # Verify manifest updates
        mock_manifest.update_status.assert_has_calls([
            call(0, Status.DONE),
            call(1, Status.DONE)
        ])

from unittest.mock import patch

def test_pipeline_error_handling():
    mock_manifest = MagicMock()
    mock_manifest.get_pending_indices.return_value = [0]
    mock_manifest.get_row.return_value = {'source_uri': 'in/bad.jpg', 'output_uri': 'out/bad.jpg'}
    
    mock_io = MagicMock()
    mock_io.read.side_effect = Exception("S3 Error")
    
    processor = BatchProcessor(MagicMock(), mock_manifest, mock_io, MagicMock(), MagicMock())
    
    processor.run_sequential()
    
    mock_manifest.update_status.assert_called_with(0, Status.FAILED, "S3 Error")

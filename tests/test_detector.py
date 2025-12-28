import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.detector import FaceDetector, BBox

def test_detector_initialization():
    with patch('mediapipe.tasks.vision.FaceDetector.create_from_options') as mock_create:
        detector = FaceDetector(model_path="src/models/dummy.tflite", device="CPU")
        mock_create.assert_called_once()

def test_detector_detect():
    with patch('mediapipe.tasks.vision.FaceDetector.create_from_options') as mock_create:
        mock_detector_instance = MagicMock()
        mock_create.return_value = mock_detector_instance
        
        # Mock result
        mock_detection = MagicMock()
        mock_bbox = MagicMock()
        mock_bbox.origin_x = 100
        mock_bbox.origin_y = 100
        mock_bbox.width = 50
        mock_bbox.height = 60
        mock_detection.bounding_box = mock_bbox
        
        mock_result = MagicMock()
        mock_result.detections = [mock_detection]
        mock_detector_instance.detect.return_value = mock_result
        
        detector = FaceDetector(model_path="src/models/dummy.tflite")
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        bboxes = detector.detect(image)
        
        assert len(bboxes) == 1
        assert bboxes[0].x == 100
        assert bboxes[0].y == 100
        assert bboxes[0].w == 50
        assert bboxes[0].h == 60

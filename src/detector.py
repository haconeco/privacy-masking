import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
import numpy as np

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

class FaceDetector:
    def __init__(self, model_path: str, device: str = 'CPU', min_detection_confidence: float = 0.5):
        delegate = python.BaseOptions.Delegate.CPU
        if device == 'GPU':
            delegate = python.BaseOptions.Delegate.GPU
        
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=delegate
        )
        
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image: np.ndarray) -> list[BBox]:
        # Convert numpy (H, W, 3) to MP Image
        # MediaPipe expects RGB. OpenCV is BGR usually. 
        # We assume input is already RGB or we should convert?
        # Usually libraries assume RGB. Caller should ensure RGB.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        detection_result = self.detector.detect(mp_image)
        
        bboxes = []
        for detection in detection_result.detections:
            # bbox is bounding_box object with origin_x, origin_y, width, height
            b = detection.bounding_box
            bboxes.append(BBox(x=b.origin_x, y=b.origin_y, w=b.width, h=b.height))
            
        return bboxes

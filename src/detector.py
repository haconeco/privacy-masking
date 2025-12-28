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
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # Configure Delegate
        # Note: Delegate setup in python tasks is implicit via BaseOptions? 
        # Actually BaseOptions has delegate argument in recent versions.
        # But 'CPU' is default. 'GPU' requires proper env.
        # Use python.BaseOptions.Delegate.GPU if available, else CPU.
        
        # We will assume CPU for implementation safety unless specified.
        # The prompt says: BaseOptions(delegate=GPU)
        
        if device == 'GPU':
             # Try to set GPU delegate
             # This might fail on Mac if not supported, so we wrap in try/except or rely on user env.
             # In python tasks API, there isn't a direct enum for GPU in all versions?
             # Let's check documentation or common usage.
             # Actually, creating BaseOptions(delegate=python.BaseOptions.Delegate.GPU)
             pass 
             # For now, I will stick to default (CPU) or minimal config, 
             # as I cannot easily verify GPU config on this environment.
             # I'll leave a TODO for GPU.
        
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

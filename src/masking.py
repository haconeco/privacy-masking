import cv2
import numpy as np
from typing import List, Tuple
from src.detector import BBox

class PrivacyMasker:
    def apply_blur(self, image: np.ndarray, bboxes: List[BBox], expand_ratio: float = 0.2, blur_kernel_ratio: float = 0.1) -> np.ndarray:
        """
        Apply Gaussian blur to the bbox regions.
        blur_kernel_ratio: kernel size relative to bbox size (e.g. 1/10 of width)
        """
        result_image = image.copy()
        h_img, w_img = image.shape[:2]
        
        for bbox in bboxes:
            expanded = self._expand_bbox(bbox, (h_img, w_img), expand_ratio)
            
            roi = result_image[expanded.y:expanded.y+expanded.h, expanded.x:expanded.x+expanded.w]
            
            if roi.size == 0:
                continue
                
            # Determine kernel size (must be odd)
            ksize = int(min(expanded.w, expanded.h) * blur_kernel_ratio)
            if ksize % 2 == 0:
                ksize += 1
            if ksize < 1:
                ksize = 1
                
            # heavy blur
            # Use GaussianBlur. SigmaX=0 means calculated from kernel size
            blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            
            result_image[expanded.y:expanded.y+expanded.h, expanded.x:expanded.x+expanded.w] = blurred_roi
            
        return result_image

    def _expand_bbox(self, bbox: BBox, img_shape: Tuple[int, int], ratio: float) -> BBox:
        h_img, w_img = img_shape
        
        # Current center
        cx = bbox.x + bbox.w / 2
        cy = bbox.y + bbox.h / 2
        
        # New size
        nw = bbox.w * (1 + ratio)
        nh = bbox.h * (1 + ratio)
        
        # New top-left
        nx = int(cx - nw / 2)
        ny = int(cy - nh / 2)
        nw = int(nw)
        nh = int(nh)
        
        # Clip
        nx = max(0, nx)
        ny = max(0, ny)
        
        # Clip width/height
        if nx + nw > w_img:
            nw = w_img - nx
        if ny + nh > h_img:
            nh = h_img - ny
            
        return BBox(x=nx, y=ny, w=nw, h=nh)

    def remove_exif(self, image_bytes: bytes) -> bytes:
        # OpenCV decode->encode pipeline naturally removes EXIF usually.
        # If we handle bytes -> decode -> process -> encode -> bytes, it is cleared.
        # This method might be a helper if we just want to strip EXIF without re-encoding (lossless),
        # but the request implies we are processing the image, so re-encoding happens anyway.
        # So essentially 'encode' is the strip step.
        pass

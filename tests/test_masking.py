import pytest
import numpy as np
import cv2
from src.masking import PrivacyMasker
from src.detector import BBox

def test_mask_blur():
    # Create white image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    # BBox at center
    bbox = BBox(x=40, y=40, w=20, h=20)
    
    masker = PrivacyMasker()
    # Mask with black color or blur? 
    # Blur on uniform color doesn't change much unless noise.
    # So let's create a random noise image.
    np.random.seed(42)
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    original_roi = image[40:60, 40:60].copy()
    
    masked_image = masker.apply_blur(image.copy(), [bbox], expand_ratio=0.0)
    
    # Check that region changed
    masked_roi = masked_image[40:60, 40:60]
    assert not np.array_equal(original_roi, masked_roi)
    
    # Check that outside didn't change (e.g., top left)
    assert np.array_equal(image[0:10, 0:10], masked_image[0:10, 0:10])

def test_bbox_expansion():
    # Test internal expansion logic
    masker = PrivacyMasker()
    bbox = BBox(x=50, y=50, w=20, h=20)
    img_shape = (100, 100)
    
    # Expand 0.5 -> 50% increase in size (10px each side if symmetric? or total?)
    # usually 1.2x means 20% larger.
    # If standard implementation: new_w = w * (1+ratio)
    
    expanded = masker._expand_bbox(bbox, img_shape, 0.5)
    
    # w=20 -> w=30 (add 5 each side)
    # x=50 -> x=45
    assert expanded.w == 30
    assert expanded.h == 30
    assert expanded.x == 45
    assert expanded.y == 45

def test_bbox_expansion_clip():
    # Test clipping at boundaries
    masker = PrivacyMasker()
    bbox = BBox(x=10, y=10, w=20, h=20)
    img_shape = (25, 25) # small image
    
    expanded = masker._expand_bbox(bbox, img_shape, 1.0) # double size -> w=40
    
    # Should clip to image bounds 0..25
    # Center approx 20, 20. new w 40. x start around 0.
    
    assert expanded.x >= 0
    assert expanded.y >= 0
    assert expanded.x + expanded.w <= 25
    assert expanded.y + expanded.h <= 25

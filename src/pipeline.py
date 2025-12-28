import cv2
import numpy as np
from dataclasses import dataclass
import traceback
import multiprocessing as mp
import ctypes
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.manifest import ManifestManager, Status
from src.io_handler import IOHandler, get_io_handler
from src.detector import FaceDetector, BBox
from src.masking import PrivacyMasker

@dataclass
class ProcessingConfig:
    expand_ratio: float = 0.2
    blur_kernel_ratio: float = 0.1
    jpeg_quality: int = 95
    num_processes: int = 4
    batch_size: int = 100

def _process_batch_item(
    idx: int,
    source_uri: str, 
    output_uri: str, 
    config: ProcessingConfig,
    model_path: str,
    device: str
) -> Tuple[int, str, str]: # idx, status, error_msg
    """
    Worker function for parallel processing.
    Instantiates its own Detector and Handlers to avoid pickling issues.
    """
    try:
        # Re-instantiate components per process
        # For IO, we need to know if it is S3 or Local. get_io_handler helper is useful.
        io_handler = get_io_handler(source_uri)
        
        # Detector
        # CAUTION: MediaPipe initialization might be heavy. 
        # In a real heavy pipeline, we might want to initialize once per worker process.
        # ProcessPoolExecutor re-uses workers, so we can use a global or cached initialization mechanism.
        
        # Simple caching using global variable
        global _detector_instance
        if '_detector_instance' not in globals() or _detector_instance is None:
            _detector_instance = FaceDetector(model_path=model_path, device=device)
        detector = _detector_instance
        
        masker = PrivacyMasker()
        
        # 1. Read
        img_bytes = io_handler.read(source_uri)
        
        # 2. Decode
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return idx, Status.FAILED.value, f"Failed to decode image: {source_uri}"
        
        # 3. Detect
        bboxes = detector.detect(image)
        
        # 4. Mask
        masked_image = masker.apply_blur(
            image, 
            bboxes, 
            expand_ratio=config.expand_ratio, 
            blur_kernel_ratio=config.blur_kernel_ratio
        )
        
        # 5. Encode
        retval, buf = cv2.imencode('.jpg', masked_image, [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality])
        if not retval:
            return idx, Status.FAILED.value, "Failed to encode image"
            
        # 6. Write (Target IO might be different from Source IO)
        out_io = get_io_handler(output_uri)
        out_io.write(output_uri, buf.tobytes())
        
        return idx, Status.DONE.value, ""
        
    except Exception as e:
        return idx, Status.FAILED.value, str(e)

# Global initializer for worker
_detector_instance = None
def _worker_init(model_path, device):
    global _detector_instance
    # Initialize detector once per worker
    try:
        _detector_instance = FaceDetector(model_path=model_path, device=device)
    except Exception as e:
        print(f"Worker init failed: {e}")

class BatchProcessor:
    def __init__(self, 
                 config: ProcessingConfig,
                 manifest: ManifestManager,
                 io_handler: IOHandler,
                 detector: FaceDetector,
                 masker: PrivacyMasker,
                 model_path: str = "src/models/face_detection_short_range.tflite",
                 device: str = "CPU"):
        self.config = config
        self.manifest = manifest
        self.io_handler = io_handler # Used in sequential
        self.detector = detector     # Used in sequential
        self.masker = masker         # Used in sequential
        
        # For parallel execution to know how to init workers
        self.model_path = model_path
        self.device = device

    def run_sequential(self):
        pending_indices = self.manifest.get_pending_indices()
        for idx in pending_indices:
            row = self.manifest.get_row(idx)
            self._process_single_sequential(idx, row['source_uri'], row['output_uri'])
        self.manifest.save()

    def _process_single_sequential(self, idx, source_uri, output_uri):
        try:
            img_bytes = self.io_handler.read(source_uri)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode: {source_uri}")
            
            bboxes = self.detector.detect(image)
            masked_image = self.masker.apply_blur(image, bboxes, self.config.expand_ratio, self.config.blur_kernel_ratio)
            
            retval, buf = cv2.imencode('.jpg', masked_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality])
            if not retval: raise ValueError("Encode failed")
            
            self.io_handler.write(output_uri, buf.tobytes())
            self.manifest.update_status(idx, Status.DONE)
        except Exception as e:
            self.manifest.update_status(idx, Status.FAILED, str(e))

    def run_parallel(self):
        pending_indices = self.manifest.get_pending_indices()
        total = len(pending_indices)
        if total == 0:
            return

        print(f"Processing {total} items with {self.config.num_processes} processes...")
        
        # We chunk the work or submit all?
        # For very large lists, submit all might consume memory?
        # futures list is small (just objects).
        
        chunk_size = 100 # save manifest every 100
        
        with ProcessPoolExecutor(max_workers=self.config.num_processes, 
                                 initializer=_worker_init, 
                                 initargs=(self.model_path, self.device)) as executor:
            
            futures = {}
            for idx in pending_indices:
                row = self.manifest.get_row(idx)
                f = executor.submit(
                    _process_batch_item, 
                    idx, 
                    row['source_uri'], 
                    row['output_uri'], 
                    self.config,
                    self.model_path,
                    self.device
                )
                futures[f] = idx
            
            completed_count = 0
            for future in as_completed(futures):
                idx, status_val, error_msg = future.result()
                status = Status(status_val)
                self.manifest.update_status(idx, status, error_msg)
                
                completed_count += 1
                if completed_count % chunk_size == 0:
                    self.manifest.save()
                    print(f"Propagated {completed_count}/{total}")
            
            self.manifest.save()
            print("Parallel processing complete.")

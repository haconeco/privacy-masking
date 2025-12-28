import argparse
import sys
import os
import glob
from src.pipeline import BatchProcessor, ProcessingConfig
from src.manifest import ManifestManager
from src.detector import FaceDetector
from src.masking import PrivacyMasker
from src.io_handler import get_io_handler
import boto3

def list_s3_objects(uri: str):
    # s3://bucket/prefix
    from urllib.parse import urlparse
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    uris = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    uris.append(f"s3://{bucket}/{key}")
    return uris

def main():
    parser = argparse.ArgumentParser(description="Privacy Masking System")
    parser.add_argument("--manifest", required=True, help="Path to manifest file (CSV/Parquet)")
    parser.add_argument("--input", help="Input path (Local dir glob or S3 prefix). Required for initialization.")
    parser.add_argument("--output", help="Output path (Local dir or S3 prefix). Required for initialization.")
    parser.add_argument("--model-path", default="src/models/face_detection_short_range.tflite", help="Path to MediaPipe model")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"], help="Inference device")
    parser.add_argument("--expand-ratio", type=float, default=0.2, help="BBox expansion ratio")
    parser.add_argument("--blur-ratio", type=float, default=0.1, help="Blur kernel size ratio")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Output JPEG quality")
    
    # Parallel options
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--num-processes", type=int, default=4, help="Number of processes for parallel execution")
    
    args = parser.parse_args()
    
    # 1. Manifest
    manifest = ManifestManager(args.manifest)
    
    if args.input and args.output:
        print(f"Initializing manifest from {args.input}...")
        sources = []
        if args.input.startswith("s3://"):
            sources = list_s3_objects(args.input)
        else:
            if os.path.isdir(args.input):
                pattern = os.path.join(args.input, "**", "*.[jJ][pP][gG]")
                sources = glob.glob(pattern, recursive=True)
                pattern_png = os.path.join(args.input, "**", "*.[pP][nN][gG]")
                sources.extend(glob.glob(pattern_png, recursive=True))
            else:
                sources = glob.glob(args.input, recursive=True)
        
        if not args.input.startswith("s3://"):
             sources = [os.path.abspath(p) for p in sources]
             
        manifest.initialize(sources, args.output)
        print(f"Manifest initialized with {len(sources)} items.")
    
    if manifest.df.empty:
        print("Manifest is empty. Please provide --input and --output to initialize.")
        sys.exit(0)
    
    class UnifiedIOHandler:
        def read(self, uri: str) -> bytes:
            return get_io_handler(uri).read(uri)
        def write(self, uri: str, data: bytes) -> None:
            get_io_handler(uri).write(uri, data)
            
    io_handler = UnifiedIOHandler()
    
    detector = FaceDetector(model_path=args.model_path, device=args.device)
    masker = PrivacyMasker()
    
    config = ProcessingConfig(
        expand_ratio=args.expand_ratio, 
        blur_kernel_ratio=args.blur_ratio,
        jpeg_quality=args.jpeg_quality,
        num_processes=args.num_processes
    )
    
    processor = BatchProcessor(
        config, manifest, io_handler, detector, masker,
        model_path=args.model_path, device=args.device
    )
    
    print(f"Starting processing (Parallel={args.parallel}, Processes={args.num_processes})...")
    if args.parallel:
        processor.run_parallel()
    else:
        processor.run_sequential()
    print("Processing complete.")

if __name__ == "__main__":
    main()

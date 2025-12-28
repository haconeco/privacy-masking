import pandas as pd
import os
from enum import Enum
from typing import List, Optional

class Status(Enum):
    PENDING = "PENDING"
    DONE = "DONE"
    FAILED = "FAILED"

class ManifestManager:
    COLUMNS = ['source_uri', 'output_uri', 'status', 'error_message']

    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.df = pd.DataFrame()
        if os.path.exists(manifest_path) and os.path.getsize(manifest_path) > 0:
            self.load()

    def load(self):
        if self.manifest_path.endswith('.parquet'):
            self.df = pd.read_parquet(self.manifest_path)
        else:
            self.df = pd.read_csv(self.manifest_path)

    def save(self):
        if self.manifest_path.endswith('.parquet'):
            self.df.to_parquet(self.manifest_path, index=False)
        else:
            self.df.to_csv(self.manifest_path, index=False)

    def initialize(self, source_uris: List[str], output_base_uri: str):
        """
        Initialize manifest from list of source URIs.
        Only adds new URIs if they don't exist? 
        For now, simplistic implementation: Overwrite or create new from scratch.
        The user requirement said "First run creates PENDING".
        """
        # If file exists, we probably loaded it. 
        # But if we want to initialize from a list, strictly speaking we might be merging.
        # For this task, let's assume we are creating fresh or appending. 
        # But the specific test case creates new.
        
        data = []
        for src in source_uris:
            # Simple filename based output mapping
            filename = os.path.basename(src)
            # Handle potential URL/URI structure roughly
            if '://' in output_base_uri:
                 # simplistic join for s3:// or file://
                 if output_base_uri.endswith('/'):
                     dst = output_base_uri + filename
                 else:
                     dst = output_base_uri + '/' + filename
            else:
                dst = os.path.join(output_base_uri, filename)

            data.append({
                'source_uri': src,
                'output_uri': dst,
                'status': Status.PENDING.value,
                'error_message': ''
            })
        
        self.df = pd.DataFrame(data)
        self.save()

    def update_status(self, index: int, status: Status, error_message: str = ''):
        # We assume index corresponds to DataFrame index.
        # In distributed processing, we might need a lock or write-ahead logging,
        # but for this assignment, direct DF update is fine as per "Single GPU machine".
        self.df.at[index, 'status'] = status.value
        self.df.at[index, 'error_message'] = error_message

    def get_pending_indices(self) -> List[int]:
        """Returns indices of items that need processing (PENDING or FAILED)"""
        if self.df.empty:
            return []
        
        # Filter for PENDING or FAILED
        mask = self.df['status'].isin([Status.PENDING.value, Status.FAILED.value])
        return self.df[mask].index.tolist()

    def get_row(self, index: int):
        return self.df.iloc[index]

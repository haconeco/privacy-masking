import pytest
import pandas as pd
import tempfile
import os
from src.manifest import ManifestManager, Status

def test_manifest_initialization():
    source_uris = ["file:///tmp/1.jpg", "file:///tmp/2.jpg"]
    output_base = "file:///tmp/out"
    
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        manager = ManifestManager(tmp.name)
        manager.initialize(source_uris, output_base)
        
        df = pd.read_csv(tmp.name)
        assert len(df) == 2
        assert all(df['status'] == Status.PENDING.value)
        assert df.iloc[0]['source_uri'] == "file:///tmp/1.jpg"
        assert df.iloc[0]['output_uri'] == "file:///tmp/out/1.jpg"

def test_manifest_update():
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        # Prep initial csv
        df = pd.DataFrame({
            'source_uri': ['s3://bucket/1.jpg'],
            'output_uri': ['s3://bucket/out/1.jpg'],
            'status': [Status.PENDING.value],
            'error_message': ['']
        })
        df.to_csv(tmp.name, index=False)
        
        manager = ManifestManager(tmp.name)
        manager.update_status(0, Status.DONE)
        manager.save()
        
        df_new = pd.read_csv(tmp.name)
        assert df_new.iloc[0]['status'] == Status.DONE.value

def test_manifest_resume():
    """Test that we can load an existing manifest and know what's pending"""
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        df = pd.DataFrame({
            'source_uri': ['1.jpg', '2.jpg', '3.jpg'],
            'output_uri': ['out/1.jpg', 'out/2.jpg', 'out/3.jpg'],
            'status': [Status.DONE.value, Status.FAILED.value, Status.PENDING.value],
            'error_message': ['', 'Error', '']
        })
        df.to_csv(tmp.name, index=False)
        
        manager = ManifestManager(tmp.name)
        pending_indices = manager.get_pending_indices()
        
        # FAILED items might also need retry depending on policy, but usually PENDING + FAILED (if retry)
        # For now let's assume get_pending_indices returns PENDING and FAILED
        assert 2 in pending_indices
        assert 1 in pending_indices # FAILED should be retried? User said "Pending/Failed only pick up"
        assert 0 not in pending_indices

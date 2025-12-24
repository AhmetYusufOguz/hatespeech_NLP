# model_size.py
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from huggingface_hub import snapshot_download

repo_id = "sentence-transformers/all-MiniLM-L6-v2"
print("Downloading / locating model:", repo_id)
path = snapshot_download(repo_id, cache_dir=None)  # default cache

print("Model path:", path)

def folder_size_mb(start_path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024*1024)

print("Total size (MB):", round(folder_size_mb(path), 2))
#!/usr/bin/env python3
"""
Download DeepSORT checkpoint file
"""

import os
import urllib.request
import sys

# ZQPei's raw GitHub path 404s (the repo stores the weights via Git LFS / Drive),
# which silently saves an HTML error page as ckpt.t7. Use a mirror that serves
# the real ~44MB Market1501 re-ID checkpoint as a release asset.
CHECKPOINT_URL = "https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/releases/download/v.2.0/ckpt.t7"
CHECKPOINT_PATH = "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"

def download_file(url, path):
    """Download a file with progress"""
    print(f"Downloading DeepSORT checkpoint...")
    print(f"URL: {url}")
    print(f"Destination: {path}")
    
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r  Progress: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, path, reporthook)
        print("\n✅ Download complete!")
        
        # Check file size
        size = os.path.getsize(path)
        print(f"   File size: {size / 1024 / 1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def main():
    print("=" * 70)
    print("📥 DeepSORT Checkpoint Downloader")
    print("=" * 70)
    
    # Check if already exists
    if os.path.exists(CHECKPOINT_PATH):
        size = os.path.getsize(CHECKPOINT_PATH)
        print(f"\n✅ Checkpoint already exists: {CHECKPOINT_PATH}")
        print(f"   Size: {size / 1024 / 1024:.1f} MB")
        
        response = input("\nRe-download? (y/n): ").strip().lower()
        if response != 'y':
            print("   Using existing checkpoint.")
            return 0
    
    # Download
    if download_file(CHECKPOINT_URL, CHECKPOINT_PATH):
        print("\n✅ DeepSORT checkpoint ready!")
        print("   The system will now use advanced tracking.")
        return 0
    else:
        print("\n⚠️  Download failed.")
        print("   The system will still work with simple detection.")
        print("   Tracking will be less accurate but functional.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

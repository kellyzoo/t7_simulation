import os
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path
import cv2

capture_path = "/Users/borabayazit/tcig_coded_sensor/REVEAL_T7-main/src/python/image/captures"
output_path = "./analysis"
os.makedirs(output_path, exist_ok=True)

PIXEL_INTENSITY_RANGE = 32  # Intensity range: 0 to 31
NUM_BINS = 32  # Bins for noise histograms
H, W = 480, 640


def compute_clean_image(image_dir, kernel_size=31, sigma=5):
    """Compute the clean image by averaging all burst images in a directory."""

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.npy")))
    bursts = [cv2.GaussianBlur(np.load(file), (kernel_size, kernel_size), sigma) for file in image_files]
    
    return np.mean(bursts, axis=0)


def compute_pmfs(capture_path):
    """Compute a noise distribution pmf per pixel location, per intensity (H x W x 32) total pmfs"""

    pmfs = np.zeros((H, W, PIXEL_INTENSITY_RANGE, NUM_BINS), dtype=np.float32)
    
    capture_path = Path(capture_path)
    exposure_dirs = sorted([d for d in capture_path.iterdir() if d.is_dir()])
    
    for exposure_dir in tqdm(exposure_dirs, desc="Processing exposures"):
        clean_dir = exposure_dir / "clean"
        
        clean_image = compute_clean_image(clean_dir)
        clean_image = (clean_image[:, 640:] * 31).astype(np.int8)
        for image_file in sorted(clean_dir.glob("*.npy")):
            burst_image = np.load(image_file)
            burst_image = (burst_image[:, 640:] * 31).astype(np.int8)
            noise = burst_image - clean_image
            for intensity in range(PIXEL_INTENSITY_RANGE):
                mask = (clean_image == intensity)
                coords = np.argwhere(mask)
                for r, c in coords:
                    pmfs[r, c, intensity, noise[r, c] + intensity] += 1    
                    
        
                    
    sums = np.sum(pmfs, axis=3, keepdims=True) + 1e-9
    pmfs /= sums
    return pmfs

pmfs = compute_pmfs(capture_path)
np.save(os.path.join(output_path, "pixel_intensity_pmfs.npy"), pmfs)

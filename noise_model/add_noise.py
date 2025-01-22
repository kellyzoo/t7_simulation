import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

capture_path = "/Users/borabayazit/tcig_coded_sensor/REVEAL_T7-main/src/python/image/captures"
output_clean_path = "./analysis/clean_images"
output_noisy_path = "./analysis/noisy_images"
pmf_path = "./analysis/pixel_intensity_pmfs.npy"

os.makedirs(output_clean_path, exist_ok=True)
os.makedirs(output_noisy_path, exist_ok=True)

PIXEL_INTENSITY_RANGE = 32  # Intensity range: 0 to 31
NUM_BINS = 32  # Bins for noise histograms
H, W = 480, 640

pmfs = np.load(pmf_path)

def compute_clean_image(image_dir):
    """Compute the clean image by averaging all burst images in a directory."""
    image_files = sorted(image_dir.glob("*.npy"))
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    bursts = [np.load(file) for file in image_files]
    clean_image = np.mean(bursts, axis=0)

    # Scale the intensities (if required)
    clean_image = (clean_image[:, 640:] * 31).astype(np.int8)
    
    return clean_image

def cdf_from_pmf(pmf):
    """Convert PMF to CDF."""
    return np.cumsum(pmf, axis=-1)

def sample_noise_from_cdf(cdf, intensity):
    """
    Perform inversion sampling to generate random samples from a given CDF.
    """
    random_values = np.random.rand(*cdf.shape[:-1]) 
    idx = np.searchsorted(cdf, random_values, side="right")
    return idx - intensity

def synthesize_noisy_image(clean_image, pmfs):
    """
    Synthesize a noisy image from a clean image using precomputed PMFs.
    """
    H, W = clean_image.shape
    noisy_image = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            intensity = int(clean_image[i, j])  
            cdf = cdf_from_pmf(pmfs[i, j, intensity])  
            noise = sample_noise_from_cdf(cdf, intensity)  
            noisy_image[i, j] = clean_image[i, j] + noise  

    return noisy_image

def process_exposures(capture_path, output_clean_path, output_noisy_path):
    capture_path = Path(capture_path)
    output_clean_path = Path(output_clean_path)
    output_noisy_path = Path(output_noisy_path)
    
    exposure_dirs = sorted([d for d in capture_path.iterdir() if d.is_dir()])

    for exposure_dir in tqdm(exposure_dirs, desc="Processing exposures"):
        clean_dir = exposure_dir / "clean"
        # Compute clean image
        clean_image = compute_clean_image(clean_dir)
        clean_image_save_path = output_clean_path / f"{exposure_dir.name}_clean.npy"
        np.save(clean_image_save_path, clean_image)
        print(f"Saved clean image to: {clean_image_save_path}")
        
        # Synthesize noisy image
        noisy_image = synthesize_noisy_image(clean_image, pmfs)
        noisy_image_save_path = output_noisy_path / f"{exposure_dir.name}_noisy.npy"
        np.save(noisy_image_save_path, noisy_image)
        print(f"Saved noisy image to: {noisy_image_save_path}")


process_exposures(capture_path, output_clean_path, output_noisy_path)
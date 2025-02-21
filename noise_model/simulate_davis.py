import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import timeit
# Paths
input_image_dir = "/Users/borabayazit/Downloads/DAVIS/JPEGImages/Full-Resolution"  # Change to your dataset directory
output_clean_dir = "./processed_davis/clean"
output_noisy_dir = "./processed_davis/noisy"
pmf_path = "/Users/borabayazit/tcig_coded_sensor/t7_simulation/noise_model/analysis/pixel_intensity_pmfs_9x9_ransac.npy"

os.makedirs(output_clean_dir, exist_ok=True)
os.makedirs(output_noisy_dir, exist_ok=True)

TARGET_HEIGHT, TARGET_WIDTH = 480, 640
PIXEL_INTENSITY_RANGE = 32  # Intensity range: 0 to 31

pmfs = np.load(pmf_path)
np.set_printoptions(precision=8, suppress=True)


def cdf_from_pmf(pmf):
    """Convert PMF to CDF."""
    return np.cumsum(pmf, axis=-1)

def sample_noise_from_cdf(cdf, intensity):
    """
    Perform inversion sampling to generate random samples from a given CDF.
    """
    rand_val = np.random.rand() 
    idx = np.searchsorted(cdf, rand_val, side="right")
    return idx - intensity

def synthesize_noisy_image(clean_image, pmfs):
    """
    Synthesize a noisy image from a clean image using precomputed PMFs, fully vectorized.
    Assumes:
      - clean_image.shape == (H, W)
      - pmfs.shape == (H, W, 32, )  # each pixel+intensity has a 1D PMF of length 32
      - clean_image intensities in [0, 31]
    """
    H, W = clean_image.shape
    
    # Flatten the clean image for vectorized indexing
    clean_flat = clean_image.ravel()  # shape: (H*W,)
    
    # Row/col indices for each pixel
    rows = np.repeat(np.arange(H), W)  # (H*W,)
    cols = np.tile(np.arange(W), H)    # (H*W,)
    
    # Gather the PMFs relevant to each pixel's clean intensity
    # pmfs[r, c, intensity] -> shape: (H*W, 32)
    pmfs_for_pixels = pmfs[rows, cols, clean_flat]
    
    # Compute the sum of each PMF to check for zero-sum cases
    pmf_sums = pmfs_for_pixels.sum(axis=1)
    
    # Convert PMFs to CDFs by cumulative summation along the last axis
    cdfs_for_pixels = np.cumsum(pmfs_for_pixels, axis=1)
    
    # Generate random values for each pixel for inversion sampling
    rand_vals = np.random.rand(H * W)
    
    # Use np.searchsorted to perform sampling on the entire array
    # idx_flat will be the sampled (noisy) intensity for each pixel
    idx_flat = np.sum(cdfs_for_pixels <= rand_vals[:, None], axis=1)
    # The actual noise added is (sampled_intensity - original_intensity)
    noise_flat = idx_flat - clean_flat
    
    # If the PMF sums to zero, we skip adding noise (i.e., keep the pixel as is)
    zero_mask = (pmf_sums == 0)
    noise_flat[zero_mask] = 0
    
    # Construct the final noisy image by adding the noise
    noisy_flat = clean_flat + noise_flat
    
    # Reshape back to (H, W)
    noisy_image = noisy_flat.reshape(H, W)
    
    return noisy_image

def process_images(input_dir, clean_output_dir, noisy_output_dir):
    input_dir = Path(input_dir)
    clean_output_dir = Path(clean_output_dir)
    noisy_output_dir = Path(noisy_output_dir)

    img_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    numimgs = 0
    for img_dir in tqdm(img_dirs, desc="Processing directories"):
        os.makedirs(os.path.join(clean_output_dir,img_dir.stem), exist_ok=True) 
        os.makedirs(os.path.join(output_noisy_dir,img_dir.stem), exist_ok=True) 
        numimgs += len(list(img_dir.glob("*.jpg")))
        
    for img_dir in tqdm(img_dirs, desc="Processing directories"): 
        image_files = list(img_dir.glob("*.jpg"))
        for image_path in tqdm(image_files, desc="Processing images"):
            # Read and preprocess image
            image = cv2.imread(str(image_path))
            
            # Resize to 480x640
            resized_image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

            # Convert to grayscale
            grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            # Normalize intensity range from 0-255 to 0-31 for model compatibility
            clean_image = (grayscale_image / 255.0 * (PIXEL_INTENSITY_RANGE - 1)).astype(np.uint8)

            # Save clean image as JPG (scaled back to 0-255 range)
            clean_image_jpg = (clean_image / (PIXEL_INTENSITY_RANGE - 1) * 255).astype(np.uint8)
            clean_image_path = clean_output_dir / img_dir.stem / f"{image_path.stem}_clean.jpg"
            cv2.imwrite(str(clean_image_path), clean_image_jpg)

            t_0 = timeit.default_timer()
            # Generate noisy image
            noisy_image = synthesize_noisy_image(clean_image, pmfs)
            t_1 = timeit.default_timer()


            # Save noisy image as JPG
            noisy_image_path = noisy_output_dir / img_dir.stem / f"{image_path.stem}_noisy.jpg"
            noisy_image_jpg = (noisy_image / (PIXEL_INTENSITY_RANGE - 1) * 255).astype(np.uint8)
            elapsed_time = round((t_1 - t_0) * 10 ** 6, 3)
            print(f"Elapsed time: {elapsed_time} µs")
            cv2.imwrite(str(noisy_image_path), noisy_image_jpg)



process_images(input_image_dir, output_clean_dir, output_noisy_dir)

print("All images processed and saved.")

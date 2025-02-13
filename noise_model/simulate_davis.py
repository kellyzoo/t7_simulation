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
pmf_path = "./analysis/pixel_intensity_pmfs.npy"

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
    Synthesize a noisy image from a clean image using precomputed PMFs.
    """
    H, W = clean_image.shape
    noisy_image = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            intensity = int(clean_image[i, j])  
            cdf = cdf_from_pmf(pmfs[i, j, intensity])
            if pmfs[i, j, intensity].sum() == 0.0:
                noisy_image[i, j] = clean_image[i, j]  
                continue
            noise = sample_noise_from_cdf(cdf, intensity)  
            noisy_image[i, j] = clean_image[i, j] + noise  

    return noisy_image

def process_images(input_dir, clean_output_dir, noisy_output_dir):
    input_dir = Path(input_dir)
    clean_output_dir = Path(clean_output_dir)
    noisy_output_dir = Path(noisy_output_dir)

    img_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    numimgs = 0
    for img_dir in tqdm(img_dirs, desc="Processing directories"): 
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
            clean_image_path = clean_output_dir / f"{image_path.stem}_clean.jpg"
            cv2.imwrite(str(clean_image_path), clean_image_jpg)

            t_0 = timeit.default_timer()
            # Generate noisy image
            noisy_image = synthesize_noisy_image(clean_image, pmfs)
            t_1 = timeit.default_timer()


            # Save noisy image as JPG
            noisy_image_path = noisy_output_dir / f"{image_path.stem}_noisy.jpg"
            noisy_image_jpg = (noisy_image / (PIXEL_INTENSITY_RANGE - 1) * 255).astype(np.uint8)
            elapsed_time = round((t_1 - t_0) * 10 ** 6, 3)
            print(f"Elapsed time: {elapsed_time} µs")
            cv2.imwrite(str(noisy_image_path), noisy_image_jpg)



process_images(input_image_dir, output_clean_dir, output_noisy_dir)

print("All images processed and saved.")

import os
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path
import cv2
from PIL import Image
import scipy

capture_path = "/Users/borabayazit/tcig_coded_sensor/REVEAL_T7-main/src/python/image/captures"
output_path = "/Users/borabayazit/tcig_coded_sensor/t7_simulation/noise_model/analysis"
os.makedirs(output_path, exist_ok=True)

PIXEL_INTENSITY_RANGE = 32  # Intensity range: 0 to 31
NUM_BINS = 32              # Bins for noise histograms
H, W = 480, 640

filtered_slopes = np.load("slope_map_raw_333x333.npy")
intercepts = np.load("intercepts_333x333npy.npy")
#filtered_slopes = scipy.ndimage.median_filter(filtered_slopes, size=245)
#intercepts = scipy.ndimage.median_filter(intercepts, size=241)

#fitted_values = np.load("fitted_values_31x31.npy")

time_indices = np.arange(150)  # Shape: (150,)
fitted_vals = filtered_slopes[:, :, np.newaxis] * time_indices + intercepts[:, :, np.newaxis]  # Shape: (H, W, 150)

def compute_clean_image(image_dir, kernel_size=333):
    """Compute the clean image by averaging all burst images in a directory."""
    bursts = []
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.npy")))
    for file in image_files:
        imgArr = np.load(file)
        # Crop or split whichever is appropriate for your data
        imgArr = imgArr[:, 640:]
        imgArr = (imgArr * 255).astype(np.uint8)
        burst = cv2.medianBlur(imgArr, ksize=kernel_size)
        bursts.append(burst)
    return np.mean(bursts, axis=0)

def compute_pmfs(capture_path):
    """
    Compute a noise distribution pmf per pixel location, per intensity:
    Shape: (H, W, 32, 32).
    pmfs[r, c, intensity, noisy_intensity] = Probability of seeing 'noisy_intensity'
    given the 'clean_image' intensity = intensity at (r, c).
    """
    pmfs = np.zeros((H, W, PIXEL_INTENSITY_RANGE, NUM_BINS), dtype=np.float32)

    capture_path = Path(capture_path)
    exposure_dirs = sorted([d for d in capture_path.iterdir() if d.is_dir()])

    clean_out_path = os.path.join(output_path, "clean_images_333x333_ransac")
    os.makedirs(clean_out_path, exist_ok=True)
    i = 0

    for exposure_dir in tqdm(exposure_dirs, desc="Processing exposures"):
        clean_dir = exposure_dir / "clean"

        # Compute “clean” reference image from burst
        #clean_image = compute_clean_image(clean_dir)
        clean_image  = fitted_vals[:,:,i]
        
        clean_image = np.clip(clean_image, min=0.0, max=1.0)
        print(clean_image)
        # Save a JPEG for inspection
        clean_img_jpeg = Image.fromarray((clean_image * 255).astype(np.uint8))
        img_save_path = os.path.join(clean_out_path, f"{exposure_dir.name}.jpeg")
        clean_img_jpeg.save(img_save_path)

        # Convert clean image to [0..31] range as int
        clean_image_31 = ((clean_image) * 31).astype(np.int16)

        # Process all burst images
        burst_images = sorted(clean_dir.glob("*.npy"))
        for image_file in tqdm(burst_images, desc=f"Bursts in {exposure_dir.name}", leave=False):
            burst_image = np.load(image_file)
            burst_image = burst_image[:, 640:]  # match the same crop as in compute_clean_image
            burst_image_31 = (burst_image * 31).astype(np.int16)

            # noise = (burst_pixel_value) - (clean_pixel_value)
            noise = burst_image_31 - clean_image_31

            # We want: pmfs[r, c, intensity, intensity + noise] += 1
            # Build advanced indices:
            intensities = clean_image_31  # shape (H, W)
            noisy_intensities = clean_image_31 + noise  # shape (H, W)

            # Valid range mask to avoid index out of bounds
            valid_mask = (
                (intensities >= 0) & (intensities < PIXEL_INTENSITY_RANGE) &
                (noisy_intensities >= 0) & (noisy_intensities < NUM_BINS)
            )

            # Get row, col coords where valid
            rows, cols = np.where(valid_mask)

            # Now, advanced indexing
            pmfs[rows, 
                 cols, 
                 intensities[rows, cols], 
                 noisy_intensities[rows, cols]] += 1

        i += 1

    # Normalize along the noise (last) dimension
    sums = np.sum(pmfs, axis=3, keepdims=True) + 1e-9
    pmfs /= sums

    return pmfs

pmfs = compute_pmfs(capture_path)
np.save(os.path.join(output_path, "pixel_intensity_pmfs_333x333_ransac.npy"), pmfs)

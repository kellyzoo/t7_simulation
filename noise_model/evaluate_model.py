import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import cv2
from scipy.spatial.distance import jensenshannon

capture_path = "/Users/borabayazit/tcig_coded_sensor/REVEAL_T7-main/src/python/image/captures"
clean_images_path = "./analysis/clean_images_333x333_filter_245"
output_comparison_path = "./analysis/comparison_results"

os.makedirs(output_comparison_path, exist_ok=True)

pmfs = np.load("/Users/borabayazit/tcig_coded_sensor/t7_simulation/noise_model/analysis/pixel_intensity_pmfs_333x333_filtered_245.npy")


def synthesize_noisy_image(clean_image):
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

def load_images_from_directory(directory):
    """Load all images from a given directory."""
    image_files = sorted(Path(directory).glob("*.npy"))
    if not image_files:
        raise ValueError(f"No images found in {directory}")
    
    images = [np.load(file) for file in image_files]
    images = np.stack(images, axis=0)
    return images

def compute_noise_pmf(noisy_images, clean_image, bins=63, noise_range=(-31.5, 31.5)):
    """
    Compute the PMF of noise, given noisy images and a clean reference.
    """
    all_noise_values = []

    for noisy_img in noisy_images:
        noise = noisy_img - clean_image  # Compute noise frame
        noise_int = noise.round().astype(np.int8)  # Ensure integer values
        all_noise_values.extend(noise_int.ravel())  # Flatten into 1D

    # Compute histogram (normalized)
    pmf, bin_edges = np.histogram(all_noise_values, bins=bins, range=noise_range, density=True)

    return pmf

def compute_js_divergence(pmf1, pmf2):
    """Compute Jensen-Shannon divergence (JSD) between two probability distributions."""
    js_distance = jensenshannon(pmf1, pmf2)
    return js_distance ** 2  # JSD = (Jensen-Shannon distance)^2

def process_comparisons(capture_path, output_comparison_path):
    capture_path = Path(capture_path)
    exposure_dirs = sorted([d for d in capture_path.iterdir() if d.is_dir()])

    jsd_all = []

    for exposure_dir in tqdm(exposure_dirs, desc="Comparing images"):
        exposure_name = exposure_dir.name
        exposure_dir = exposure_dir / "clean"
        clean_image_f = clean_images_path + f"/{exposure_name}.jpeg"
        clean_image = cv2.imread(clean_image_f, cv2.IMREAD_GRAYSCALE)
        clean_image = ((clean_image / 255) * 31).astype(np.int8)  # Scale to [0, 1] floats
        real_noisy_images = load_images_from_directory(exposure_dir)
        real_noisy_images = (real_noisy_images[:,:, 640:] * 31).astype(np.int8)
        synth_noisy_images = []
        for i in tqdm(range(50)):
            synth_noisy_images.append(synthesize_noisy_image(clean_image))

        pmf_real = compute_noise_pmf(real_noisy_images, clean_image)
        pmf_synth = compute_noise_pmf(synth_noisy_images, clean_image)

        jsd_value = compute_js_divergence(pmf_real, pmf_synth)
        jsd_all.append(jsd_value)

        print(f"Exposure: {exposure_name} - JSD: {jsd_value:.6f}")
    
    print(f"\nMean JSD across exposures: {np.mean(jsd_all):.6f}")
process_comparisons(capture_path, output_comparison_path)
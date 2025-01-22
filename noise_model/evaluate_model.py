import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

capture_path = "/Users/borabayazit/tcig_coded_sensor/REVEAL_T7-main/src/python/image/captures"
noisy_image_path = "./analysis/noisy_images"
output_comparison_path = "./analysis/comparison_results"

os.makedirs(output_comparison_path, exist_ok=True)

def load_images_from_directory(directory):
    """Load all images from a given directory."""
    image_files = sorted(Path(directory).glob("*.npy"))
    if not image_files:
        raise ValueError(f"No images found in {directory}")
    
    images = [np.load(file) for file in image_files]
    return images

def compare_images(original_images, noisy_image):
    """Compare the synthesized noisy image with original captured images."""
    mse_scores = []
    psnr_scores = []
    ssim_scores = []

    for original_image in original_images:
        original_image = (original_image[:, 640:] * 31).astype(np.int8)
        
        # Use three different metrics to compare the simulated noisy image to actual captures: MSE, PSNR, SSIM
        mse = mean_squared_error(original_image, noisy_image)
        psnr = peak_signal_noise_ratio(original_image, noisy_image, data_range=31)
        ssim = structural_similarity(original_image, noisy_image, data_range=31)

        mse_scores.append(mse)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

    # Obtain the average scores over all captures
    return np.mean(mse_scores), np.mean(psnr_scores), np.mean(ssim_scores)

def process_comparisons(capture_path, noisy_image_path, output_comparison_path):
    capture_path = Path(capture_path)
    noisy_image_path = Path(noisy_image_path)

    exposure_dirs = sorted([d for d in capture_path.iterdir() if d.is_dir()])

    mse_all = []
    psnr_all = []
    ssim_all = []

    comparison_results = []

    for exposure_dir in tqdm(exposure_dirs, desc="Comparing images"):
        exposure_name = exposure_dir.name
        noisy_image_file = noisy_image_path / f"{exposure_name}_noisy.npy"

        if noisy_image_file.exists():
            original_images = load_images_from_directory(exposure_dir)
            noisy_image = np.load(noisy_image_file)

            mse, psnr, ssim = compare_images(original_images, noisy_image)
            comparison_results.append([exposure_name, mse, psnr, ssim])

            mse_all.append(mse)
            psnr_all.append(psnr)
            ssim_all.append(ssim)

            print(f"Comparison for {exposure_name}: MSE={mse:.2f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}")

    mean_mse = np.mean(mse_all)
    mean_psnr = np.mean(psnr_all)
    mean_ssim = np.mean(ssim_all)

    print("\nOverall Mean Metrics Across All Exposures:")
    print(f"Mean MSE: {mean_mse:.2f}")
    print(f"Mean PSNR: {mean_psnr:.2f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")

process_comparisons(capture_path, noisy_image_path, output_comparison_path)
import numpy as np
import os
from pathlib import Path
from pillow_heif import register_heif_opener
from PIL import Image

register_heif_opener()  # Enables Pillow to open HEIC images

# -----------------------------------------------------------------
# 1) HELPER FUNCTIONS
# -----------------------------------------------------------------

def load_heic_as_gray_and_resize(
    image_path,
    width=640,
    height=480
):
    """
    Opens a HEIC image, converts to grayscale, and resizes to (width, height).
    """
    with Image.open(image_path) as im:
        # Convert to grayscale (8-bit L mode)
        gray_im = im.convert("L")
        
        # Resize to the target dimensions (width, height)
        gray_resized = gray_im.resize((width, height), resample=Image.BICUBIC)
        
        # Convert to NumPy array
        arr = np.array(gray_resized, dtype=np.float32)
        
        # Scale from 0–255 float to 0–31 int
        arr_0to31 = np.round(arr / 255.0 * 31.0).astype(np.int8)
        return arr_0to31


def cdf_from_pmf(pmf):
    """Convert a 1D PMF to a 1D CDF."""
    return np.cumsum(pmf, axis=-1)


def sample_noise_from_cdf(cdf, intensity):
    """
    Inversion sampling to generate random noise from a given CDF.
    """
    rand_val = np.random.rand() 
    idx = np.searchsorted(cdf, rand_val, side="right")
    return idx - intensity


def synthesize_noisy_image(clean_image, pmfs):
    """
    Synthesize a noisy image from a clean image using precomputed PMFs.
    pmfs is assumed to be shaped like [H, W, 32], or [H, W, 32, ...] 
    such that pmfs[i, j, intensity] is the 1D PMF for that pixel & intensity.
    """
    H, W = clean_image.shape
    noisy_image = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            intensity = int(clean_image[i, j])  
            pmf_ij = pmfs[i, j, intensity]  # shape: (32,)
            
            # If PMF is all zeros, keep the original intensity
            if pmf_ij.sum() == 0.0:
                noisy_image[i, j] = intensity
                continue

            cdf = cdf_from_pmf(pmf_ij)
            noise = sample_noise_from_cdf(cdf, intensity)
            noisy_image[i, j] = intensity + noise  

    return noisy_image


# -----------------------------------------------------------------
# 2) MAIN SCRIPT
# -----------------------------------------------------------------

if __name__ == "__main__":
    # Paths
    heic_image_path = "/Users/borabayazit/Downloads/IMG_2550.HEIC"  # <--- Change this to your HEIC file
    pmf_path = "./analysis/pixel_intensity_pmfs_333x333_ransac.npy"  # Path to your precomputed PMF array
    output_noisy_path = "my_photo_noisy.png"

    # Load PMFs (shape might be e.g. [480, 640, 32])
    pmfs = np.load(pmf_path)

    # Load and crop the HEIC image to 480×640
    clean_img_0to31 = load_heic_as_gray_and_resize(
        heic_image_path,
    )

    # Add noise
    noisy_img = synthesize_noisy_image(clean_img_0to31, pmfs)

    # --------------------------------------------------------------
    # Optionally, save the noisy image for quick viewing as PNG
    # Map [0..31] back to [0..255] 
    # --------------------------------------------------------------
    noisy_img_clamped = np.clip(noisy_img, 0, 31)
    # Scale from [0..31] → [0..255]
    noisy_img_8bit = (noisy_img_clamped / 31.0 * 255.0).astype(np.uint8)

    # Save as a PNG
    noisy_pil = Image.fromarray(noisy_img_8bit, mode="L")
    noisy_pil.save(output_noisy_path)
    print("Noisy image saved to:", output_noisy_path)

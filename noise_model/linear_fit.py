import os
import numpy as np
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from tqdm import tqdm

# For parallelization
from joblib import Parallel, delayed

# Directory containing JPEG images
dir_path = "./analysis/clean_images_333x333"

# Load all images (grayscale)
file_list = sorted([f for f in os.listdir(dir_path) if f.endswith(".jpeg")])
images_list = [
    cv2.imread(os.path.join(dir_path, f), cv2.IMREAD_GRAYSCALE) 
    for f in file_list
]
images = np.stack(images_list, axis=0)  # Shape: (num_images, height, width)
images = (images / 255.0).astype(np.float32)  # Scale to [0, 1] floats

num_images, height, width = images.shape

# Flatten images from shape (num_images, height, width)
# to shape (height*width, num_images) for easier parallelization:
pixel_traces = images.reshape(num_images, -1).T  # shape => (n_pixels, n_times)

# Define the piecewise function: linear up to x_break, then constant.
def piecewise_linear(x, x_break, m, b, c):
    return np.where(x < x_break, m * x + b, c)

# Function to fit a single pixel's time trace:
def fit_pixel(pixel_values, X_vals, initial_guess, param_bounds):
    """
    pixel_values: 1D array of shape (num_images,) with pixel intensities over time
    X_vals: 1D array of time indices
    initial_guess, param_bounds: passed directly to curve_fit

    Returns:
        popt = best-fit parameters [x_break, m, b, c]
        r2   = coefficient of determination
        fitted_curve = piecewise values over X_vals
    """
    try:
        popt, _ = curve_fit(
            piecewise_linear,
            X_vals,
            pixel_values,
            p0=initial_guess,
            bounds=param_bounds,
            maxfev=5000
        )
        fitted_curve = piecewise_linear(X_vals, *popt)
        corr, _ = pearsonr(pixel_values, fitted_curve)
        r2 = corr**2
    except RuntimeError:
        # If fitting fails
        popt = [0.0, 0.0, 0.0, 0.0]  # or any fallback
        r2 = 0.0
        fitted_curve = np.zeros_like(pixel_values)
    return popt, r2, fitted_curve

# Precompute the 'time' vector
X_vals = np.arange(num_images).astype(float)

# Initial guess for curve_fit
initial_guess = [num_images / 2, 0.0, 0.0, 1.0]

# Optional: param bounds
param_bounds = (
    [0.0, -np.inf, -np.inf, 0.0],         # lower
    [float(num_images), np.inf, np.inf, 2.0]  # upper
)

# Run the fits in parallel with joblib
# n_jobs=-1 uses all available cores; adjust if needed
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(fit_pixel)(pixel_traces[i], X_vals, initial_guess, param_bounds)
    for i in range(pixel_traces.shape[0])
)

# `results` is a list of tuples (popt, r2, fitted_curve) for each pixel
# We now unpack into separate arrays

# Create arrays to store parameters
t_break_map   = np.zeros((height * width,), dtype=np.float32)
slope_map     = np.zeros((height * width,), dtype=np.float32)
intercept_map = np.zeros((height * width,), dtype=np.float32)
sat_map       = np.zeros((height * width,), dtype=np.float32)
r2_map        = np.zeros((height * width,), dtype=np.float32)
fitted_vals_all = np.zeros((height * width, num_images), dtype=np.float32)

for i, (popt, r2, fitted_curve) in enumerate(results):
    x_break_fit, m_fit, b_fit, c_fit = popt
    t_break_map[i]   = x_break_fit
    slope_map[i]     = m_fit
    intercept_map[i] = b_fit
    sat_map[i]       = c_fit
    r2_map[i]        = r2
    fitted_vals_all[i, :] = fitted_curve

# Reshape all param maps from (height*width,) to (height, width)
t_break_map   = t_break_map.reshape(height, width)
slope_map     = slope_map.reshape(height, width)
intercept_map = intercept_map.reshape(height, width)
sat_map       = sat_map.reshape(height, width)
r2_map        = r2_map.reshape(height, width)
fitted_vals_all = fitted_vals_all.reshape(height, width, num_images)

# Optional: median filter on slope_map
filtered_slope_map = scipy.ndimage.median_filter(slope_map, size=5)

# Save the arrays if needed
np.save("pwlf_t_break_map.npy", t_break_map)
np.save("pwlf_slope_map.npy", slope_map)
np.save("pwlf_intercept_map.npy", intercept_map)
np.save("pwlf_sat_map.npy", sat_map)
np.save("pwlf_r2_map.npy", r2_map)
np.save("pwlf_fitted_values.npy", fitted_vals_all)

print("Average R² across all pixels:", np.mean(r2_map))

# Visualize some results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Fitted Breakpoint (x_break)")
plt.imshow(t_break_map, interpolation='nearest')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Slope (m) - raw")
plt.imshow(slope_map, interpolation='nearest')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("R² Score")
plt.imshow(r2_map, interpolation='nearest')
plt.colorbar()

plt.tight_layout()
plt.savefig("piecewise_linear_fit_results.png", dpi=150)
plt.show()

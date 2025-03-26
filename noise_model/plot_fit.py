import os
import numpy as np
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sklearn
from sklearn.linear_model import RANSACRegressor, LinearRegression
from tqdm import tqdm
# Directory containing JPEG images
dir_path = "./analysis/clean_images_333x333"

# Load all images (grayscale)
file_list = sorted([f for f in os.listdir(dir_path) if f.endswith(".jpeg")])
images = [cv2.imread(os.path.join(dir_path, f), cv2.IMREAD_GRAYSCALE) for f in file_list]
images = np.stack(images, axis=0)  # Shape: (num_images, height, width)
images = (images / 255).astype(np.float32)  # Scale to [0, 1] floats

num_images, height, width = images.shape


# Choose a pixel to inspect. For example, use the center pixel:
pixel_y, pixel_x = 150, 200

# Extract the actual pixel values over time
pixel_values = images[:, pixel_y, pixel_x]

slopes =  np.load("pwlf_slope_map.npy")
intercepts =  np.load("pwlf_intercept_map.npy")
#slopes = scipy.ndimage.median_filter(slopes, size=245)


plt.subplot(1, 3, 2)
plt.title("Filtered Slopes")
plt.imshow(slopes, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.show()

# Extract the fitted values from the RANSAC model for this pixel
time_indices = np.arange(150)  # Shape: (150,)
fitted_vals = slopes[:, :, np.newaxis] * time_indices + intercepts[:, :, np.newaxis]  # Shape: (H, W, 150)
fitted_values = fitted_vals[pixel_y, pixel_x, :]  # Shape: (150,)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(np.arange(num_images), pixel_values, 'bo-', label='Actual Data')
plt.plot(np.arange(num_images), fitted_values, 'r--', label='Linear Fit')
plt.xlabel("Frame (Time)")
plt.ylabel("Pixel Intensity")
plt.title(f"Pixel ({pixel_y}, {pixel_x}) - Actual vs Fitted Data")
plt.legend()
plt.show()

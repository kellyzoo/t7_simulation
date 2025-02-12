import numpy as np
import cv2
from utils import _write_as_png, _generate_test_image
import torch
import torch.nn.functional as F

import os
import argparse
from pathlib import Path

def create_filter(K, position):
    i, j = position // K, position % K
    # Reverse the vertical position (K-1 - i instead of i)
    i = K - 1 - i
    j = K - 1 - j
    filter = np.zeros((K, K))
    filter[i, j] = 1
    return filter

def reshuffle_mosaic2vid(image, K):
    """
    Extract frames from mosaic using convolution with KxK filters.
    
    Args:
    - image (numpy array): Input mosaic image of shape (H, W)
    - K (int): Size of each KxK tile
    - bucket (int): 0 for left bucket, 1 for right bucket
    
    Returns:
    - frames (numpy array): Extracted frames of shape (K^2, H//K, W//K)
    """
    H, W = image.shape
    
    # Compute the padding required to make H and W divisible by K
    pad_h = (K - H % K) % K
    pad_w = (K - W % K) % K
    
    # Pad the image with zeros
    if pad_h != 0 or pad_w != 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    # Reshape the image to extract KxK tiles

    # Get output dimensions
    out_H, out_W = image.shape[0] // K, image.shape[1] // K
    frames = np.zeros((K * K, out_H, out_W))
    
    # Create all K^2 frames using convolution
    for pos in range(K * K):
        # Create filter for this position
        kernel = create_filter(K, pos)
        image_tensor = torch.from_numpy(image).float().reshape(1, 1, *image.shape)
        kernel_tensor = torch.from_numpy(kernel).float().reshape(1, 1, *kernel.shape)
        frame = F.conv2d(image_tensor, kernel_tensor, stride=K)
        frames[pos] = frame.numpy().squeeze()

    frames = np.concatenate([frames[K:], frames[:K]], axis=0)    

    return frames



if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        type=str,
        help='Path to the mosaic image',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the output folder',
        required=True
    )

    args = parser.parse_args()
    image_path = args.image_path
    output_dir = args.output_dir

    K = 16
    # # Presume 320 x 640 image
    image = cv2.imread(image_path, 0)
    image = (image.astype(float) * 16)
    # image = np.tile(_generate_test_image(K, 320), (1, 2)) * 256 * 16
    print(f"Input image shape: {image.shape}")
    # Each bucket individually
    size = image.shape[1] // 2
    image_0 = image[:, :size]
    image_1 = image[:, size:]

    # _write_as_png(f"./outputs/frame.png", image)
    _write_as_png(os.path.join(output_dir, f"{Path(image_path).stem}_left.png"), image_0)
    _write_as_png(os.path.join(output_dir, f"{Path(image_path).stem}_right.png"), image_1)

    frames_0 = reshuffle_mosaic2vid(image_0, K)
    frames_1 = reshuffle_mosaic2vid(image_1, K)

    # print(f"Reshuffled frames shape: {frames_0.shape}")
    # print(f"Reshuffled frames shape: {frames.shape}")

    # # Save the frames as .npy and .png files
    # Show base images
    file_name = __file__.split("/")[-1].split(".")[0]
    for i, frame in enumerate(frames_0):
        # The output folder in inverse_solvers
        # Resize
        # frame = cv2.resize(frame, (frame.shape[1] * (K), frame.shape[0] * (K)), interpolation=cv2.INTER_LINEAR)
        _write_as_png(os.path.join(output_dir, f"{Path(image_path).stem}_left_{i:05d}.png"), frame)
        # _write_as_png(f"./outputs/{file_name}_frame_0_{i:05d}.png", frame)
    for i, frame in enumerate(frames_1):
        # The output folder in inverse_solvers
        # Resize
        # frame = cv2.resize(frame, (frame.shape[1] * (K), frame.shape[0] * (K)), interpolation=cv2.INTER_LINEAR)
        _write_as_png(os.path.join(output_dir, f"{Path(image_path).stem}_right_{i:05d}.png"), frame)
        # _write_as_png(f"./outputs/{file_name}_frame_1_{i:05d}.png", frame)

import numpy as np
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--subframes',
        type=int,
        help='Number of subframes. MUST be a square number',
        required=True
    )
    parser.add_argument(
        '--frame_size',
        type=int,
        help='Size of the frame',
        required=False,
        default=320
    )

    return parser.parse_args()

def get_pattern(K):
    # Initialize an empty list to hold the patterns
    patterns = []
    
    # Generate K^2 patterns
    for i in range(K):
        for j in range(K):
            # Create a K x K matrix filled with zeros
            pattern = np.zeros((K, K), dtype=int)
            # Set the (i, j)-th element to 1
            pattern[i, j] = 1
            # Append this pattern to the list
            patterns.append(pattern)
    
    # Convert the list of patterns to a 3D numpy array of shape (K^2, K, K)
    patterns_array = np.array(patterns)
    
    return patterns_array


if __name__ == '__main__':
    args = parse_args()

    # subframes must be a square number
    tile_size = int(np.sqrt(args.subframes))
    img_size = args.frame_size

    patterns = get_pattern(tile_size)

    mask = np.zeros((args.subframes, img_size, img_size), dtype=np.uint8)
    tile_multiplier = img_size // tile_size + 1 # we cut off the extra pixels later

    for i in range(args.subframes):
        pattern = patterns[i]
        mask[i] = np.tile(pattern, (tile_multiplier, tile_multiplier))[:img_size, :img_size]
    
    mask = mask.reshape(args.subframes * img_size, img_size)
    mask *= 255

    cv2.imwrite(f'masks/coded_exposure_{tile_size}x{tile_size}.bmp', mask)

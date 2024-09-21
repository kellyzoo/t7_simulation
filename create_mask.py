import numpy as np
import cv2

if __name__ == '__main__':
    subframes = 4

    patterns = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]])

    mask = np.zeros((subframes, 320, 320), dtype=np.uint8)

    for i in range(subframes):
        pattern = patterns[i]
        mask[i] = np.tile(pattern, (160, 160)) # (160, 160)
    
    mask = mask.reshape(subframes * 320, 320)
    mask *= 255

    cv2.imwrite('masks/coded_exposure.bmp', mask)

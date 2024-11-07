import os
import cv2
import subprocess
from tqdm import tqdm
from pathlib import Path

coded_dir = '/Users/borabayazit/Downloads/X4K1000FPS_coded'
folders = sorted(os.listdir(coded_dir))
if '.DS_Store' in folders:
    folders.remove('.DS_Store')

for folder in tqdm(folders):
    subfolders = sorted(os.listdir(os.path.join(coded_dir, folder)))
    if '.DS_Store' in subfolders:
        subfolders.remove('.DS_Store')
    
    for subfolder in tqdm(subfolders):
        groups = sorted(os.listdir(os.path.join(coded_dir, folder, subfolder)))
        if '.DS_Store' in groups:
            groups.remove('.DS_Store')
        for group in groups:
            group_dir = os.path.join(coded_dir, folder, subfolder, group)
            frames = sorted(os.listdir(group_dir))

            if '.DS_Store' in frames:
                frames.remove('.DS_Store')

            # Only keep files with .png extension
            frames = sorted([frame for frame in frames if frame.endswith('.png')])
            print(f"Processing {subfolder}_{group} with {len(frames)} frames")

            for i, frame in enumerate(frames):
                # open the image
                img_path = os.path.join(group_dir, frame)
                img = cv2.imread(img_path, 0) # 0 for grayscale
                cv2.imwrite(os.path.join(group_dir, f"{subfolder}_{group}_frame_{i:05d}.png"), img)

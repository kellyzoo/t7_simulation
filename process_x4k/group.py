import os
from tqdm import tqdm
import shutil
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description="Group frames into subdirectories based on K value.")
parser.add_argument(
    "K",
    type=int,
    help="Number of frames to group together."
)

args = parser.parse_args()
K = args.K

# Get list of files in the NfS directory
train_dir = "/Users/borabayazit/Downloads/X4K1000FPS/train"
train_folders = os.listdir(train_dir)
if ".DS_Store" in train_folders:
    train_folders.remove('.DS_Store')

coded_dir =  "/Users/borabayazit/Downloads/X4K1000FPS_coded"

t7_height = 480
t7_width = 640

for folder in tqdm(train_folders):
    subfolders = sorted(os.listdir(os.path.join(train_dir, folder)))
    if '.DS_Store' in subfolders:
        subfolders.remove('.DS_Store')
    # os.makedirs(os.path.join(nfs_dir, file, str(frame_rate), 'grouped'), exist_ok=True)

    # split frames into groups of K frames (discard the last few frames if necessary)
    
    for subfolder in tqdm(subfolders):
        frames = sorted(os.listdir(os.path.join(train_dir, folder, subfolder)))
        if '.DS_Store' in frames:
            frames.remove('.DS_Store')
        num_frames = len(frames)
        num_groups = num_frames // K  

        for group_index in range(num_groups):
            group = [frames[i] for i in range(group_index, num_frames, num_groups)]
            group = group[:K]

            group_dir = os.path.join(coded_dir, folder, subfolder, f'{group_index}')
            os.makedirs(group_dir, exist_ok=True)

            for frame in group:
                frame_path = os.path.join(train_dir, folder, subfolder, frame)
                
                # Open the image and print its resolution
                with Image.open(frame_path) as img:
                    width, height = img.size
                    left = (width - t7_width) / 2
                    top = (height - t7_height) / 2
                    right = (width + t7_width) / 2
                    bottom = (height + t7_height) / 2

                    cropped_img = img.crop((left, top, right, bottom))
                    cropped_img.save(os.path.join(group_dir, frame))


import os
import subprocess
from tqdm import tqdm

coded_dir = '/Users/borabayazit/Downloads/X4K1000FPS_coded'
folders = sorted(os.listdir(coded_dir))
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
            coded_img_path = os.path.join(group_dir, f'{subfolder}_{group}.png')

            # # Delete the output_fname if it exists
            # outputs = [os.path.join(group_dir, f'{file}_{group}_left.png'),
            #            os.path.join(group_dir, f'{file}_{group}_right.png'),
            #            os.path.join(group_dir, f'{file}_{group}_left_00.png'),
            #            os.path.join(group_dir, f'{file}_{group}_left_01.png'),
            #            os.path.join(group_dir, f'{file}_{group}_left_02.png'),
            #            os.path.join(group_dir, f'{file}_{group}_left_03.png'),
            #            os.path.join(group_dir, f'{file}_{group}_right_00.png'),
            #            os.path.join(group_dir, f'{file}_{group}_right_01.png'),
            #            os.path.join(group_dir, f'{file}_{group}_right_02.png'),
            #            os.path.join(group_dir, f'{file}_{group}_right_03.png'),]

            # for output_fname in outputs:    
            #     if os.path.exists(output_fname):
            #         os.remove(output_fname)

            # Run the reshuffle script
            print(f"Running reshuffle script on {coded_img_path}")
            subprocess.run(['python', 'inverse_solvers/reshuffle.py',
                            '--image_path', coded_img_path,
                            '--output_dir', group_dir])


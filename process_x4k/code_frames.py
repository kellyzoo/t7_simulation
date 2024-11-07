import os
import subprocess
from tqdm import tqdm
from time import time

coded_dir = '/Users/borabayazit/Downloads/X4K1000FPS_coded'
folders = sorted(os.listdir(coded_dir))
if '.DS_Store' in folders:
    folders.remove('.DS_Store')

start_time = time()
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
            output_fname = os.path.join(group_dir, f'{subfolder}_{group}')

            subprocess.run(['python', 'simulate.py', 
                            '--params', 'final_T7_params.mat',
                            '--mask', 'masks/t6_coded_exposure_2x2.bmp',
                            '--input_imgs', f"{group_dir}/?.png",
                            '--output_fname', output_fname,
                            '--mode', 'multi_in_single_out',
                            '--cam_type', 't7',])

end_time = time()
duration = end_time - start_time
print(f"Execution Time: {duration:.2f} seconds")

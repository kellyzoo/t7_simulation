import os
import imageio

image_folder = "./analysis/clean_images_333x333_filter_245"
out_path = "./analysis/out.gif"

# Collect and sort all JPG/JPEG files in the image folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".jpeg")]
images.sort()

# Read each image with imageio and append to a list
frames = []
for image_name in images:
    img_path = os.path.join(image_folder, image_name)
    frames.append(imageio.imread(img_path))

# Create the GIF from the frames
# fps=5 means 5 frames per second, adjust as desired
imageio.mimsave(out_path, frames, fps=5)

print("GIF saved to:", out_path)

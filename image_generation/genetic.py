import random
import cv2
import numpy as np
import concurrent.futures
from pathlib import Path
from PIL import Image
from utils import load_img_from_url


img_url = "https://cdn.mos.cms.futurecdn.net/8pbgXKXWWZBryyVG9zABRf-1200-80.jpg"
img_url = "https://media.istockphoto.com/id/1251493047/vector/apple-flat-style-vector-icon.jpg?s=612x612&w=0&k=20&c=90r0D6OEq1wNRnfHHftEWibKBjdGe9OgMl6HEpi4d8Q="
assets_path = Path("./assets/")


def load_assets(path):
    assets = []

    for file_path in path.iterdir():
        if file_path.is_file():
            asset = Image.open(file_path)
            assets.append(asset)

    return assets


def get_compare_score(img1, img2):
    """Returns a score indicating how similar two images are"""
    return np.sum(abs(np.array(img1).flatten() - np.array(img2).flatten()))


def generate_frame(img, prev_frame: Image.Image, assets):
    """Generate a new frame based on the old frame and an list of assets"""
    new_frame = prev_frame.copy()

    random_asset = assets[random.randint(0, len(assets)-1)]

    random_angle = random.randint(0, 360)

    random_size_x = random.randint(10, 200)
    random_size_y = random.randint(10, 200)

    random_pos_x = random.randint(0, prev_frame.size[0])
    random_pos_y = random.randint(0, prev_frame.size[1])

    rotated_asset = np.array(random_asset.resize((random_size_x, random_size_y)).rotate(random_angle, expand=True))
    
    x_end = min(random_pos_x + rotated_asset.shape[1], img.width)
    y_end = min(random_pos_y + rotated_asset.shape[0], img.height)

    rotated_asset = rotated_asset[:(y_end-random_pos_y), :(x_end-random_pos_x), :]

    not_transparent_mask = np.array(rotated_asset)[:, :, 3] > 0

    img_sliced = np.array(img)[random_pos_y:y_end, random_pos_x:x_end, :]
 
    pixels = img_sliced[not_transparent_mask]

    mean_r = np.mean(pixels[:, 0])
    mean_g = np.mean(pixels[:, 1])
    mean_b = np.mean(pixels[:, 2])

    rotated_asset[not_transparent_mask, 0] = mean_r
    rotated_asset[not_transparent_mask, 1] = mean_g
    rotated_asset[not_transparent_mask, 2] = mean_b
    # rotated_asset[not_transparent_mask, 0] = random.randint(0, 255)
    # rotated_asset[not_transparent_mask, 1] = random.randint(0, 255)
    # rotated_asset[not_transparent_mask, 2] = random.randint(0, 255)

    rotated_asset = Image.fromarray(rotated_asset)

    new_frame.paste(rotated_asset, (random_pos_x, random_pos_y), rotated_asset)
    
    return new_frame


def multithread_generate_frames(img, prev_frame, assets, num_frames):
    results = []
    
    # Using ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit multiple tasks to the thread pool
        futures = [executor.submit(generate_frame, img, prev_frame, assets) for i in range(num_frames)]

        # Collect results as they are completed
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append((future.result(), get_compare_score(img, future.result())))
            except Exception as exc:
                print(f"Generated frame failed with: {exc}")
    
    return results


def generate_video(img, assets, num_frames, tries_per_frame, output_file="output_video.mp4", fps=30):
    """Generate a video of an image gradually appearing from a list of assets"""

    # Create a blank image to start with
    curr_img = Image.new('RGB', img.size, (0, 0, 0))

    # Define video codec and create VideoWriter object
    frame_size = img.size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    for i in range(num_frames): 
        # Multithreaded frame generation
        new_frames = multithread_generate_frames(img, curr_img, assets, tries_per_frame)
        best_score = 1e100

        # Select the best frame based on score
        for frame, score in new_frames:
            if score < best_score:
                curr_img = frame
                best_score = score

        # Convert PIL Image to a format suitable for OpenCV (BGR format)
        frame_bgr = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)

        # Write the frame to the video file
        out.write(frame_bgr)

        print(f"{i} / {num_frames} frames processed.")

    # Release the video writer object
    out.release()

    print(f"Video saved as {output_file}")

    return curr_img
    

img = load_img_from_url(img_url)
assets = load_assets(assets_path)

new_img = generate_video(img, assets, 100, 10)

new_img.show()

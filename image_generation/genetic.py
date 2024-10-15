import random
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
from utils import load_img_from_url


img_url = "https://cdn.mos.cms.futurecdn.net/8pbgXKXWWZBryyVG9zABRf-1200-80.jpg"
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


def generate_frame(img, prev_frame: Image.Image, assets, num_entities_tried):
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

    rotated_asset = Image.fromarray(rotated_asset)

    new_frame.paste(rotated_asset, (random_pos_x, random_pos_y), rotated_asset)
    
    return new_frame


def generate_video(img, assets, num_frames, num_comparisons):
    """Generate a video of an image gradualy appearing from a list of assets"""

    new_img = Image.new('RGB', img.size, (0, 0, 0))

    for i in range(num_frames):
        print(i)
        best_score = float("inf")
        best_frame = img
        for j in range(num_comparisons):
            new_frame = generate_frame(img, new_img, assets, 1)
            score = abs(get_compare_score(new_frame, img))
            if  score < best_score:
                best_score = score
                best_frame = new_frame
        new_img = best_frame

    return new_img



img = load_img_from_url(img_url)
black_image = Image.new('RGB', img.size, (0, 0, 0))


diff = get_compare_score(img, black_image)

assets = load_assets(assets_path)

# new_image = generate_frame(img, black_image, assets, 0)

new_image = generate_video(img, assets, 100, 100)

new_image.show()

# img.show()

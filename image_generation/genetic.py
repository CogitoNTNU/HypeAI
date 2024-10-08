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
            print(f'File: {file_path.name}')

    return assets


def get_compare_score(img1, img2):
    """Returns a score indicating how similar two images are"""
    return np.sum(abs(np.array(img1).flatten() - np.array(img2).flatten()))


def generate_frame(prev_frame: Image.Image, assets, num_entities_tried):
    """Generate a new frame based on the old frame and an list of assets"""
    new_frame = prev_frame.copy()

    random_asset = assets[random.randint(0, len(assets)-1)]

    random_angle = random.randint(0, 360)

    random_size_x = random.randint(0, 200)
    random_size_y = random.randint(0, 200)

    random_pos_x = random.randint(0, prev_frame.size[0])
    random_pos_y = random.randint(0, prev_frame.size[1])

    rotated_asset = random_asset.resize((random_size_x, random_size_y)).rotate(random_angle, expand=True)

    rotated_asset.convert('RGBA')
    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    pixels = rotated_asset.load()
    
    for y in range(rotated_asset.height):
        for x in range(rotated_asset.width):
            pixels[x][y] = (random_color,255)

    new_frame.paste(rotated_asset, (random_pos_x, random_pos_y), rotated_asset)
    
    return new_frame


def generate_video(img, assets, num_frames):
    """Generate a video of an image gradualy appearing from a list of assets"""



img = load_img_from_url(img_url)
black_image = Image.new('RGB', img.size, (0, 0, 0))


diff = get_compare_score(img, black_image)

assets = load_assets(assets_path)

print(diff)

new_image = generate_frame(black_image, assets, 0)

new_image.show()

# img.show()

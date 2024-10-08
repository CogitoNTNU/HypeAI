import requests
from PIL import Image
from pathlib import Path
from io import BytesIO
from utils import load_img_from_url

# Create assets folder
assets_path = Path("assets")
assets_path.mkdir()
print(f"Directory '{assets_path}' created!")


with open("assets.txt", "r") as file:
    for i, line in enumerate(file.readlines(), 1):
        img = load_img_from_url(line.rstrip("\n"))
        
        img_file_path = assets_path / f"image_{i}.png"
        img.save(img_file_path)

        print(f"Image {i} saved at {img_file_path}")

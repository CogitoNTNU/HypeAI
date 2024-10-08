import requests
from PIL import Image
from io import BytesIO


img_url = "https://cdn.mos.cms.futurecdn.net/8pbgXKXWWZBryyVG9zABRf-1200-80.jpg"
assets_path = "assets.txt"


def load_img_from_url(url):
    response = requests.get(url)
    img_data = response.content

    img = Image.open(BytesIO(img_data))
    return img


def load_assets(path):
    """Returns a list containing all assets from a file"""
    return True


def get_compare_score(img1, img2):
    """Returns a score indicating how similar two images are"""
    return True


def generate_frame(prev_frame, assets, num_entities_tried):
    """Generate a new frame based on the old frame and an list of assets"""


def generate_video(img, assets, num_frames):
    """Generate a video of an image gradualy appearing from a list of assets"""


img = load_img_from_url(img_url)

img.show()

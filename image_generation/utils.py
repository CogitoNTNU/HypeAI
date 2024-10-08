import requests
from PIL import Image
from io import BytesIO


def load_img_from_url(url):
    response = requests.get(url)
    img_data = response.content

    img = Image.open(BytesIO(img_data))
    return img



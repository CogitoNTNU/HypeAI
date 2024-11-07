import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from utils import load_img_from_url


img_url = "https://cdn.mos.cms.futurecdn.net/8pbgXKXWWZBryyVG9zABRf-1200-80.jpg"
assets_path = Path("./assets/")


def img_to_tensor(img: Image) -> tf.Tensor:
    return tf.convert_to_tensor(np.array(img))


def tensor_to_img(tensor: tf.Tensor) -> Image.Image:
    return Image.fromarray(tensor.numpy())


def load_assets_with_padding(path: Path) -> list[tf.Tensor]:
    assets: list[tf.Tensor] = []
    max_height = 0
    max_width = 0

    # First pass: load assets and calculate max width and height
    for file_path in path.iterdir():
        if file_path.is_file():
            img = Image.open(file_path).convert("RGBA")
            asset = img_to_tensor(img)
            assets.append(asset)
            max_height = max(max_height, asset.shape[0])
            max_width = max(max_width, asset.shape[1])


    # Second pass: pad each asset to the maximum size
    padded_assets: list[tf.Tensor] = []
    for asset in assets:
        # Calculate required padding on each side
        height_diff = max_height - asset.shape[0]
        width_diff = max_width - asset.shape[1]

        pad_top = height_diff // 2
        pad_bottom = height_diff - pad_top
        pad_left = width_diff // 2
        pad_right = width_diff - pad_left

        # Apply padding around the image
        padded_asset = tf.pad(
            asset,
            paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode='CONSTANT',
            constant_values=0  # Pads with zeros (transparent if alpha channel is present)
        )
        padded_assets.append(padded_asset)

    return padded_assets


def create_tile_batch(tensor: tf.Tensor, size: int) -> tf.Tensor:
    pass


def rotate_tensor_image_batch(images, angles):
    """Rotate a batch of image tensors by arbitrary angles (in radians) around their centers."""
    # Ensure images and angles are in the correct shape
    batch_size = tf.shape(images)[0]
    image_height = tf.cast(tf.shape(images)[1], tf.float32)
    image_width = tf.cast(tf.shape(images)[2], tf.float32)

    # Compute the center of each image
    image_center_y = image_height / 2.0
    image_center_x = image_width / 2.0

    # Compute rotation components for each angle in the batch
    cos_angles = tf.cos(-angles)
    sin_angles = tf.sin(-angles)

    # Create transformation matrices for each image in the batch
    translate_to_origin = tf.tile(tf.constant([[1, 0, -image_center_x], [0, 1, -image_center_y], [0, 0, 1]], dtype=tf.float32)[tf.newaxis, :, :], [batch_size, 1, 1])
    rotation = tf.stack([
        tf.stack([cos_angles, -sin_angles, tf.zeros_like(cos_angles)], axis=1),
        tf.stack([sin_angles, cos_angles, tf.zeros_like(sin_angles)], axis=1),
        tf.constant([0, 0, 1], dtype=tf.float32)[tf.newaxis, :]
    ], axis=1)
    translate_back = tf.tile(tf.constant([[1, 0, image_center_x], [0, 1, image_center_y], [0, 0, 1]], dtype=tf.float32)[tf.newaxis, :, :], [batch_size, 1, 1])

    # Combine transformations for each image in the batch
    combined_matrix = tf.linalg.matmul(tf.linalg.matmul(translate_back, rotation), translate_to_origin)

    # Extract the 2x3 affine matrix for each image in the batch
    affine_matrices = combined_matrix[:, :2, :]

    # Reshape affine matrices to be compatible with the ImageProjectiveTransformV2 function
    transformation_matrices = tf.concat([tf.reshape(affine_matrices, [batch_size, 6]), tf.zeros([batch_size, 2])], axis=1)
    transformation_matrices = tf.reshape(transformation_matrices, [batch_size, 8])

    # Apply affine transformations to each image in the batch
    rotated_images = tf.raw_ops.ImageProjectiveTransformV2(
        images=images,
        transforms=transformation_matrices,
        output_shape=tf.shape(images)[1:3],
        interpolation="BILINEAR"
    )

    return rotated_images


assets = load_assets_with_padding(assets_path)

tensor_img = img_to_tensor(load_img_from_url(img_url))
img = tensor_to_img(tensor_img)


img.show()

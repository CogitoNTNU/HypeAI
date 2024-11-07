import tensorflow as tf
import numpy as np
import copy
from PIL import Image
from pathlib import Path
from utils import load_img_from_url

BATCH_SIZE = 200
MIN_SCALE = 0.1
MAX_SCALE = 5

IMG_URL = "https://cdn.mos.cms.futurecdn.net/8pbgXKXWWZBryyVG9zABRf-1200-80.jpg"
ASSETS_PATH = Path("./assets/")


def img_to_tensor(img: Image) -> tf.Tensor:
    return tf.convert_to_tensor(np.array(img))


def tensor_to_img(tensor: tf.Tensor) -> Image.Image:
    # Convert tensor to numpy array
    array = tensor.numpy()

    # Ensure array is in uint8 format
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)

    # Determine the mode based on the number of channels
    if array.ndim == 2:
        mode = 'L'  # Grayscale
    elif array.shape[2] == 1:
        mode = 'L'  # Grayscale
        array = array.squeeze(axis=2)
    elif array.shape[2] == 3:
        mode = 'RGB'
    elif array.shape[2] == 4:
        mode = 'RGBA'
    else:
        raise ValueError(f"Unsupported number of channels: {array.shape[2]}")

    return Image.fromarray(array, mode)


def load_assets_with_padding(path: Path, width: int, height: int) -> tf.Tensor:
    assets: list[tf.Tensor] = []

    # First pass: load assets and calculate max width and height
    for file_path in path.iterdir():
        if file_path.is_file():
            img = Image.open(file_path).convert("RGBA")
            asset = img_to_tensor(img)
            assets.append(asset)


    # Second pass: pad each asset to the maximum size
    padded_assets: list[tf.Tensor] = []
    for asset in assets:
        # Calculate required padding on each side
        height_diff = height - asset.shape[0]
        width_diff = width - asset.shape[1]

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

    return tf.stack(padded_assets, axis=0)


@tf.function
def create_tile_batch(tensor: tf.Tensor, size: int) -> tf.Tensor:
    tensor = tf.expand_dims(tensor, axis=0)
    
    # Tile the image along the batch dimension
    batch_tensor = tf.tile(tensor, [size, 1, 1, 1])

    return batch_tensor


@tf.function
def rotate_tensor_image_batch(images: tf.Tensor, angles: tf.Tensor) -> tf.Tensor:
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

    # Prepare tensors for broadcasting
    zeros = tf.zeros([batch_size], dtype=tf.float32)
    ones = tf.ones([batch_size], dtype=tf.float32)

    # Create translation to origin matrices
    translate_to_origin = tf.transpose(
        tf.stack([
            tf.stack([ones, zeros, -image_center_x * ones], axis=0),
            tf.stack([zeros, ones, -image_center_y * ones], axis=0),
            tf.stack([zeros, zeros, ones], axis=0)
        ], axis=0),
        perm=[2, 0, 1]
    )  # Shape: [batch_size, 3, 3]

    # Create rotation matrices
    rotation = tf.transpose(
        tf.stack([
            tf.stack([cos_angles, -sin_angles, zeros], axis=0),
            tf.stack([sin_angles, cos_angles, zeros], axis=0),
            tf.stack([zeros, zeros, ones], axis=0)
        ], axis=0),
        perm=[2, 0, 1]
    )  # Shape: [batch_size, 3, 3]

    # Create translation back matrices
    translate_back = tf.transpose(
        tf.stack([
            tf.stack([ones, zeros, image_center_x * ones], axis=0),
            tf.stack([zeros, ones, image_center_y * ones], axis=0),
            tf.stack([zeros, zeros, ones], axis=0)
        ], axis=0),
        perm=[2, 0, 1]
    )  # Shape: [batch_size, 3, 3]

    # Combine transformations for each image in the batch
    transform = tf.linalg.matmul(tf.linalg.matmul(translate_back, rotation), translate_to_origin)

    # Extract the 2x3 affine matrix for each image in the batch
    affine_matrices = transform[:, :2, :]  # Shape: [batch_size, 2, 3]

    # Flatten affine matrices to shape [batch_size, 6]
    affine_matrices_flat = tf.reshape(affine_matrices, [batch_size, 6])

    # Append [0, 0] to each affine matrix to make it 8 elements
    zeros_2 = tf.zeros([batch_size, 2], dtype=tf.float32)
    transformation_matrices = tf.concat([affine_matrices_flat, zeros_2], axis=1)  # Shape: [batch_size, 8]

    # Apply affine transformations to each image in the batch
    rotated_images = tf.raw_ops.ImageProjectiveTransformV2(
        images=images,
        transforms=transformation_matrices,
        output_shape=tf.shape(images)[1:3],
        interpolation="BILINEAR"
    )

    return rotated_images


@tf.function
def scale_tensor_image_batch(images: tf.Tensor, scales: tf.Tensor) -> tf.Tensor:
    """Scale a batch of image tensors by arbitrary factors around their centers."""
    # Ensure images and scales are in the correct shape
    batch_size = tf.shape(images)[0]
    image_height = tf.cast(tf.shape(images)[1], tf.float32)
    image_width = tf.cast(tf.shape(images)[2], tf.float32)

    # Compute the center of each image
    image_center_y = image_height / 2.0
    image_center_x = image_width / 2.0

    # Prepare tensors for broadcasting
    zeros = tf.zeros([batch_size], dtype=tf.float32)
    ones = tf.ones([batch_size], dtype=tf.float32)

    # Create translation to origin matrices
    translate_to_origin = tf.transpose(
        tf.stack([
            tf.stack([ones, zeros, -image_center_x * ones], axis=0),
            tf.stack([zeros, ones, -image_center_y * ones], axis=0),
            tf.stack([zeros, zeros, ones], axis=0)
        ], axis=0),
        perm=[2, 0, 1]
    )  # Shape: [batch_size, 3, 3]

    # Create scaling matrices
    scales_inv = 1 / scales  # Invert scales because image transformation uses inverse mapping
    scaling = tf.transpose(
        tf.stack([
            tf.stack([scales_inv, zeros, zeros], axis=0),
            tf.stack([zeros, scales_inv, zeros], axis=0),
            tf.stack([zeros, zeros, ones], axis=0)
        ], axis=0),
        perm=[2, 0, 1]
    )  # Shape: [batch_size, 3, 3]

    # Create translation back matrices
    translate_back = tf.transpose(
        tf.stack([
            tf.stack([ones, zeros, image_center_x * ones], axis=0),
            tf.stack([zeros, ones, image_center_y * ones], axis=0),
            tf.stack([zeros, zeros, ones], axis=0)
        ], axis=0),
        perm=[2, 0, 1]
    )  # Shape: [batch_size, 3, 3]

    # Combine transformations for each image in the batch
    transform = tf.linalg.matmul(tf.linalg.matmul(translate_back, scaling), translate_to_origin)

    # Extract the 2x3 affine matrix for each image in the batch
    affine_matrices = transform[:, :2, :]  # Shape: [batch_size, 2, 3]

    # Flatten affine matrices to shape [batch_size, 6]
    affine_matrices_flat = tf.reshape(affine_matrices, [batch_size, 6])

    # Append [0, 0] to each affine matrix to make it 8 elements
    zeros_2 = tf.zeros([batch_size, 2], dtype=tf.float32)
    transformation_matrices = tf.concat([affine_matrices_flat, zeros_2], axis=1)  # Shape: [batch_size, 8]

    # Apply affine transformations to each image in the batch
    scaled_images = tf.raw_ops.ImageProjectiveTransformV2(
        images=images,
        transforms=transformation_matrices,
        output_shape=tf.shape(images)[1:3],
        interpolation="BILINEAR"
    )

    return scaled_images


@tf.function
def translate_tensor_image_batch(images: tf.Tensor, translations: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(images)[0]

    # Prepare tensors for broadcasting
    zeros = tf.zeros([batch_size], dtype=tf.float32)
    ones = tf.ones([batch_size], dtype=tf.float32)

    # Extract dx and dy translations and cast to float32
    dx = tf.cast(translations[:, 0], tf.float32)
    dy = tf.cast(translations[:, 1], tf.float32)

    # Create translation matrices
    translation_matrices = tf.transpose(
        tf.stack([
            tf.stack([ones, zeros, -dx], axis=0),
            tf.stack([zeros, ones, -dy], axis=0),
            tf.stack([zeros, zeros, ones], axis=0)
        ], axis=0),
        perm=[2, 0, 1]
    )  # Shape: [batch_size, 3, 3]

    # Since we're only translating, the transformation is just the translation matrix
    # Extract the 2x3 affine matrix for each image in the batch
    affine_matrices = translation_matrices[:, :2, :]  # Shape: [batch_size, 2, 3]

    # Flatten affine matrices to shape [batch_size, 6]
    affine_matrices_flat = tf.reshape(affine_matrices, [batch_size, 6])

    # Append [0, 0] to each affine matrix to make it 8 elements
    zeros_2 = tf.zeros([batch_size, 2], dtype=tf.float32)
    transformation_matrices = tf.concat([affine_matrices_flat, zeros_2], axis=1)  # Shape: [batch_size, 8]

    # Apply affine transformations to each image in the batch
    translated_images = tf.raw_ops.ImageProjectiveTransformV2(
        images=images,
        transforms=transformation_matrices,
        output_shape=tf.shape(images)[1:3],
        interpolation='BILINEAR'
    )

    return translated_images


@tf.function
def apply_avg_color_to_masked_regions(images_with_alpha: tf.Tensor, images_rgb: tf.Tensor) -> tf.Tensor:
    """
    For each image in the batch, replace the RGB values in the regions where alpha > 0
    with the average color from the corresponding regions in images_rgb.
    """
    # Ensure images are of type float32
    images_with_alpha = tf.cast(images_with_alpha, dtype=tf.float32)
    images_rgb = tf.cast(images_rgb, dtype=tf.float32)

    # Ensure the images have the same dimensions
    # assert images_with_alpha.shape[:3] == images_rgb.shape[:3], "Image dimensions must match"

    batch_size, height, width = images_with_alpha.shape[:3]

    # Create the mask where alpha > 0
    mask = images_with_alpha[..., 3] > 0  # Shape: [batch_size, height, width]
    mask = tf.cast(mask, dtype=images_rgb.dtype)  # Convert to same dtype as images
    mask = tf.expand_dims(mask, axis=-1)  # Shape: [batch_size, height, width, 1]

    # Apply mask to images_rgb to get the masked pixels
    masked_rgb = images_rgb * mask  # Shape: [batch_size, height, width, 3]

    # Sum over masked pixels
    sum_rgb = tf.reduce_sum(masked_rgb, axis=[1, 2])  # Shape: [batch_size, 3]

    # Count of masked pixels per image
    mask_counts = tf.reduce_sum(mask, axis=[1, 2])  # Shape: [batch_size, 1]

    # Compute average color safely (avoiding division by zero)
    avg_rgb = tf.math.divide_no_nan(sum_rgb, mask_counts)  # Shape: [batch_size, 3]

    # Expand avg_rgb to match image dimensions
    avg_rgb_expanded = tf.reshape(avg_rgb, [batch_size, 1, 1, 3])  # Shape: [batch_size, 1, 1, 3]
    avg_rgb_tiled = tf.broadcast_to(avg_rgb_expanded, [batch_size, height, width, 3])  # Shape: [batch_size, height, width, 3]

    # Create mask for RGB channels
    mask_rgb = tf.broadcast_to(mask, [batch_size, height, width, 3])  # Shape: [batch_size, height, width, 3]

    # Replace RGB values in images_with_alpha where mask is True
    new_rgb = tf.where(mask_rgb > 0, avg_rgb_tiled, images_with_alpha[..., :3])

    # Combine new RGB values with original alpha channel
    alpha_channel = images_with_alpha[..., 3:]  # Shape: [batch_size, height, width, 1]
    output_image = tf.concat([new_rgb, alpha_channel], axis=-1)  # Shape: [batch_size, height, width, 4]

    return output_image


@tf.function
def overlay_images_batch(background: tf.Tensor, overlays: tf.Tensor) -> tf.Tensor:
    # Expand background to match the batch size of overlays
    batch_size = tf.shape(overlays)[0]
    background_batch = tf.broadcast_to(background, [batch_size, *background.shape])

    # Split overlays into RGB and alpha channels
    overlay_rgb = overlays[..., :background.shape[-1]]
    alpha_channel = overlays[..., -1:] / 255.0  # Normalize alpha to [0, 1] range if in 8-bit

    # Blend overlay RGB with background using alpha
    result = (alpha_channel * overlay_rgb) + ((1 - alpha_channel) * background_batch)

    return tf.cast(result, background.dtype)


@tf.function
def select_images_by_indices(tensor_batch: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    
    # Gather images based on the indexes
    selected_images = tf.gather(tensor_batch, indices, axis=0)
    
    return selected_images


@tf.function
def get_compare_score(curr_tensor: tf.Tensor, ref_tensor: tf.Tensor) -> tf.Tensor:
    # Ensure both tensors are in the same integer data type
    curr_tensor_rgb = tf.cast(curr_tensor[..., :3], tf.int32)  # Only take RGB channels
    ref_tensor = tf.cast(ref_tensor, tf.int32)

    # Calculate the absolute difference and sum over all pixel values for each image in the batch
    compare_scores = tf.reduce_sum(tf.abs(curr_tensor_rgb - ref_tensor), axis=[1, 2, 3])

    # Output is a 1D tensor of shape [batch_size] with one score per image pair
    return compare_scores


class Object():
    def __init__(self, min_x: int, max_x: int, min_y: int, max_y: int, min_scale: float, max_scale: float, index_range: int):
        self.x = np.random.randint(min_x, max_x)
        self.y = np.random.randint(min_y, max_y)
        self.angle = np.random.uniform(0, 2*np.pi)
        self.scale = np.random.uniform(min_scale, max_scale)
        self.index = np.random.randint(0, index_range)


    def mutate(self, grandness_of_translation_mutation: int, grandness_of_angle_mutation: float, grandess_of_scale_mutation: float):
        self.x += np.random.randint(-grandness_of_translation_mutation, grandness_of_translation_mutation)
        self.y += np.random.randint(-grandness_of_translation_mutation, grandness_of_translation_mutation)
        self.angle += np.random.uniform(-grandness_of_angle_mutation, grandness_of_angle_mutation)
        self.scale += np.random.uniform(-grandess_of_scale_mutation, grandess_of_scale_mutation)



def generate_random_population(size: int, min_x: int, max_x: int, min_y: int, max_y: int, min_scale: float, max_scale: float, index_range: int) -> list[Object]:
    return [Object(min_x, max_x, min_y, max_y, min_scale, max_scale, index_range) for i in range(size)]


def generate_population_imgs(population: list[Object], assets: tf.Tensor, curr_img: tf.Tensor, ref_img: tf.Tensor) -> tf.Tensor:
    population_indices = tf.constant([obj.index for obj in population])
    population_translations = tf.constant([[obj.x, obj.y] for obj in population])
    population_angles = tf.constant([obj.angle for obj in population])
    population_scales = tf.constant([obj.scale for obj in population])

    population_assets = tf.gather(assets, population_indices)
    population_assets = rotate_tensor_image_batch(population_assets, population_angles)
    population_assets = scale_tensor_image_batch(population_assets, population_scales)
    population_assets = translate_tensor_image_batch(population_assets, population_translations)
    population_assets = apply_avg_color_to_masked_regions(population_assets, ref_img)

    next_images_batch = overlay_images_batch(curr_img, population_assets)

    return next_images_batch



def update_population(population: list[Object], assets: tf.Tensor, curr_img: tf.Tensor, ref_img: tf.Tensor, num_survivors: int, num_children: int, grandness_of_translation_mutation: int, grandness_of_angle_mutation: float, grandness_of_scale_mutation: float):
    next_images_batch = generate_population_imgs(population, assets, curr_img, ref_img)

    compare_scores = get_compare_score(next_images_batch, ref_img)

    survivor_indices = tf.argsort(compare_scores, direction='ASCENDING')[:num_survivors]

    survivors: list[Object] = [population[i] for i in survivor_indices]

    for i in range(num_survivors):
        for j in range(num_children):
            child = copy.deepcopy(survivors[i])
            child.mutate(grandness_of_translation_mutation, grandness_of_angle_mutation, grandness_of_scale_mutation)
            population.append(child)
    
    return survivors


def get_best_fit(population: list[Object], assets: tf.Tensor, curr_img: tf.Tensor, ref_img: tf.Tensor):
    next_images_batch = generate_population_imgs(population, assets, curr_img, ref_img)

    compare_scores = get_compare_score(next_images_batch, ref_img)

    best_index = tf.argsort(compare_scores, direction='ASCENDING')[0]

    return next_images_batch[best_index]



tensor_img = img_to_tensor(load_img_from_url(IMG_URL))
tensor_batch = create_tile_batch(tensor_img, BATCH_SIZE)

height, width = tensor_img.shape[:2]
min_x = -width//2
max_x = width//2
min_y = -height//2
max_y = height//2

curr_tensor_img = tf.zeros([height, width, 3], dtype=tf.float32)

assets = load_assets_with_padding(ASSETS_PATH, width, height)


for i in range(10):
    print(f"{i} / 10")
    population = generate_random_population(BATCH_SIZE, min_x, max_x, min_y, max_y, MIN_SCALE, MAX_SCALE, len(assets))
    curr_tensor_img = get_best_fit(population, assets, curr_tensor_img, tensor_img)



img = tensor_to_img(curr_tensor_img)
img.show()

import random
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
from utils import load_img_from_url


img_url = "https://cdn.mos.cms.futurecdn.net/8pbgXKXWWZBryyVG9zABRf-1200-80.jpg"
assets_path = Path("./assets/")


def img_to_tensor(img: Image) -> tf.Tensor:
    return tf.convert_to_tensor(np.array(img))


def tensor_to_img(tensor: tf.Tensor) -> Image.Image:
    return Image.fromarray(tensor.numpy())


def load_assets(path: Path) -> list[tf.Tensor]:
    assets = []
    for file_path in path.iterdir():
        if file_path.is_file():
            asset = img_to_tensor(Image.open(file_path).convert("RGBA"))
            assets.append(asset)
    return assets


def rotate_tensor_image(image, angle):
    """Rotate an image tensor by an arbitrary angle (in radians) around its center."""
    # Get the image dimensions and cast them to float32
    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)
    image_center_y = image_height / 2.0
    image_center_x = image_width / 2.0

    # Create the 3x3 rotation matrix around the center
    cos_angle = tf.cos(-angle)  # Negative to match standard image rotation direction
    sin_angle = tf.sin(-angle)
    
    # Transformation matrices
    translate_to_origin = [[1, 0, -image_center_x],
                           [0, 1, -image_center_y],
                           [0, 0, 1]]
    rotation = [[cos_angle, -sin_angle, 0],
                [sin_angle, cos_angle, 0],
                [0, 0, 1]]
    translate_back = [[1, 0, image_center_x],
                      [0, 1, image_center_y],
                      [0, 0, 1]]
    
    # Combine transformations: translate to origin -> rotate -> translate back
    combined_matrix = tf.linalg.matmul(tf.linalg.matmul(translate_back, rotation), translate_to_origin)
    
    # Extract the 2x3 affine matrix by removing the last row
    affine_matrix = combined_matrix[:2, :]

    # Flatten to create a 1x8 matrix
    transformation_matrix = tf.concat([tf.reshape(affine_matrix, [6]), tf.constant([0, 0], dtype=tf.float32)], axis=0)
    transformation_matrix = tf.reshape(transformation_matrix, [1, 8])

    # Apply the affine transformation
    rotated_image = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(image, 0),
        transforms=transformation_matrix,
        output_shape=tf.shape(image)[:2],
        interpolation="BILINEAR"
    )
    
    return tf.squeeze(rotated_image, 0)


class Object():
    def __init__(self, min_x: int, max_x: int, min_y: int, max_y: int, min_scale: int, max_scale: int, assets: list[tf.Tensor]):
        self.x = random.randint(min_x, max_x)
        self.y = random.randint(min_y, max_y)
        self.index = random.randint(0, len(assets)-1)
        self.angle = random.uniform(0, 2*np.pi)
        scale = random.uniform(min_scale, max_scale)
        self.width = round(assets[self.index].shape[1]*scale)
        self.height = round(assets[self.index].shape[0]*scale)
        self.assets = assets
        print(self.x)
        print(self.y)
        print(self.width)
        print(self.height)


    def place_on_tensor(self, tensor: tf.Tensor, ref_tensor: tf.Tensor):
        asset = self.assets[self.index]
        
        resized_asset = tf.image.resize(asset, [self.height, self.width])

        rotated_asset = rotate_tensor_image(resized_asset, self.angle)

        target_height, target_width, channels = tensor.shape

        # Calculate the visible area within the bounds
        tgt_x_start = tf.maximum(0, self.x)
        tgt_y_start = tf.maximum(0, self.y)
        tgt_x_end = tf.minimum(target_width, self.x + self.width)
        tgt_y_end = tf.minimum(target_height, self.y + self.height)

        # Check for overlap: If no overlap, return the original tensor
        if tgt_x_start >= tgt_x_end or tgt_y_start >= tgt_y_end:
            return tensor

        # Calculate the corresponding crop of the rotated asset
        src_x_start = tf.maximum(0, -self.x)
        src_y_start = tf.maximum(0, -self.y)
        src_x_end = src_x_start + (tgt_x_end - tgt_x_start)
        src_y_end = src_y_start + (tgt_y_end - tgt_y_start)

        # Crop the rotated asset to fit within the target bounds
        cropped_asset = rotated_asset[src_y_start:src_y_end, src_x_start:src_x_end, :]

        # Ensure tensor_area_to_update has the same data type as cropped_asset
        tensor_area_to_update = tf.cast(
            tensor[tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end, :channels], cropped_asset.dtype
        )

        # Extract the RGB and alpha channels from `cropped_asset`
        rgb_cropped_asset = cropped_asset[..., :3]
        alpha_channel = cropped_asset[..., 3:] / 255.0  # Normalize alpha to [0, 1]

        # Create a mask for non-transparent pixels, broadcasting it to match RGB channels
        non_transparent_mask = alpha_channel > 0  # Shape: [H, W, 1]
        non_transparent_mask = tf.broadcast_to(non_transparent_mask, tf.shape(rgb_cropped_asset))  # Shape: [H, W, 3]

        # Flatten spatial dimensions but keep channels
        ref_area = ref_tensor[tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end, :channels]
        ref_area_flat = tf.reshape(ref_area, [-1, channels])
        mask_flat = tf.reshape(non_transparent_mask, [-1, channels])

        # Apply the mask to get the masked area with RGB channels
        masked_area = tf.boolean_mask(ref_area_flat, mask_flat[..., 0])

        # Compute the average color (shape: [3])
        avg_color = tf.reduce_mean(masked_area, axis=0)

        # Reshape avg_color for broadcasting (shape: [1, 1, 3])
        avg_color = tf.reshape(avg_color, [1, 1, channels])

        # Fill `rgb_cropped_asset` with the average color
        rgb_cropped_asset = tf.broadcast_to(avg_color, tf.shape(rgb_cropped_asset))

        # **Cast rgb_cropped_asset to float32 for blending**
        rgb_cropped_asset = tf.cast(rgb_cropped_asset, tf.float32)

        # **Cast tensor_area_to_update to float32 for blending**
        tensor_area_to_update = tf.cast(tensor_area_to_update, tf.float32)

        # **Perform blending**
        updated_area = (alpha_channel * rgb_cropped_asset) + ((1 - alpha_channel) * tensor_area_to_update)

        # **Cast `updated_area` back to the tensor’s dtype (likely uint8)**
        updated_area = tf.cast(updated_area, tensor.dtype)

        # Prepare indices and updates for `tf.tensor_scatter_nd_update`
        indices_y = tf.range(tgt_y_start, tgt_y_end)
        indices_x = tf.range(tgt_x_start, tgt_x_end)
        indices = tf.stack(tf.meshgrid(indices_y, indices_x, indexing='ij'), axis=-1)
        indices = tf.reshape(indices, [-1, 2])

        # Flatten `updated_area` to match the structure of indices
        flat_updated_area = tf.reshape(updated_area, [-1, channels])

        # Update the tensor by placing the updated area back into it
        target = tf.tensor_scatter_nd_update(tensor, indices, flat_updated_area)

        return target


    def get_compare_score(self, curr_tensor: tf.Tensor, ref_tensor: tf.Tensor):
        tf.reduce_sum(tf.abs(tf.cast(curr_tensor, tf.int32) - tf.cast(ref_tensor, tf.int32)))



assets = load_assets(assets_path)

ref_img_tensor = img_to_tensor(load_img_from_url(img_url))
curr_img_tensor = tf.zeros(ref_img_tensor.shape, dtype=tf.uint8)

min_x = -100
min_y = -100
max_x = curr_img_tensor.shape[1]
max_y = curr_img_tensor.shape[0]
min_scale = 0.1
max_scale = 5

for i in range(100):
    obj = Object(min_x, max_x, min_y, max_y, min_scale, max_scale, assets)
    curr_img_tensor = obj.place_on_tensor(curr_img_tensor, ref_img_tensor)

final_img = tensor_to_img(curr_img_tensor)

final_img.show()

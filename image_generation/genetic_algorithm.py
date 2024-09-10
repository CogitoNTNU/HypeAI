import cv2
import numpy as np
from PIL import Image
import random

# Parameters for the video
output_video = "output.mp4"
fps = 30
duration = 5  # in seconds
num_frames = fps * duration
i = 1

# Load the image
image_path = "lilla2.jpg"  # Replace with your image
image = Image.open(image_path)
image_np = np.array(image)

# Get image dimensions
height, width, _ = image_np.shape

# Create video writer object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Initialize a completely black canvas
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Generate random shapes and sizes that progressively get smaller
for frame_count in range(num_frames):
    # Calculate the current size of shapes
    max_shape_size = max(1, int(35 - (frame_count / num_frames) * 35))

    # Number of shapes to add in this frame
    num_shapes = int(i*(200 - (frame_count / num_frames) * 180))
    i += 0.5

    # Generate shapes
    for _ in range(num_shapes):
        # Pick a random position on the image
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        # Pick the color from the original image at the selected location
        color = tuple(int(c) for c in image_np[y, x])

        # Pick a random shape size
        size = random.randint(1, max_shape_size)

        # Draw the shape (circle in this case) on the canvas
        cv2.circle(canvas, (x, y), size, color, -1)

    # Write the frame to the video
    out.write(canvas)

# Release the video writer
out.release()

print("Video generation complete!")


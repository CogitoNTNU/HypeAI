from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Function to create text image with a specific font
def create_text_image(text, font_path, font_size, text_color, size, bg_color):
    img = Image.new('RGBA', size, bg_color)  # Create a transparent background image
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)  # Load custom font
    except OSError as e:
        print(f"Error loading font: {e}")
        return None
    
    # Calculate text size using getbbox (returns the bounding box of the text)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]  # width
    h = bbox[3] - bbox[1]  # height
    
    position = ((size[0] - w) // 2, (size[1] - h) // 2)  # Center the text
    draw.text(position, text, fill=text_color, font=font)
    return img

# Load video
video = VideoFileClip("output_video_with_countdown.mp4")

# Corrected font path
font_path = "/System/Library/Fonts/SFNSRounded.ttf"

# Create text image with custom font
text_image = create_text_image(
    text="IQ score",
    font_path=font_path,  # Correct path to DejaVu Sans font
    font_size=60, 
    text_color=(255, 255, 255, 255),  # White text
    size=(500, 300),  # Width x Height of the text box
    bg_color=(0, 0, 0, 0)  # Transparent background
)

if text_image:
    # Convert the PIL image to a NumPy array
    text_image_np = np.array(text_image)

    # Convert NumPy array to MoviePy ImageClip
    text_clip = ImageClip(text_image_np).set_duration(video.duration).set_position(("center", "bottom"))

    # Composite the text onto the video
    final_clip = CompositeVideoClip([video, text_clip])

    # Export the final video
    final_clip.write_videofile("output_video_with_text.mp4", codec="libx264", fps=24)
else:
    print("Text image creation failed due to font loading issue.")

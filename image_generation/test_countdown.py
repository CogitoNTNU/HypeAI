from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips
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

# Load the original video
video = VideoFileClip("output.mp4")

# Split the video into smaller clips (1 second each for countdown)
clip_duration = 1/5  # Duration for each clip (in seconds)
num_clips = int(video.duration // clip_duration)  # Number of clips
clips = []

# Corrected font path
font_path = "/System/Library/Fonts/SFNSRounded.ttf"

# Create individual clips with countdown numbers overlaid
for i in range(num_clips, 0, -1):
    # Extract a subclip from the original video
    subclip = video.subclip((num_clips - i) * clip_duration, (num_clips - i + 1) * clip_duration)

    # Create text image for countdown
    text_image = create_text_image(
        text=str(i),  # Countdown number
        font_path=font_path,
        font_size=60,
        text_color=(255, 255, 255, 255),  # White text
        size=(500, 200),  # Width x Height of the text box
        bg_color=(0, 0, 0, 0)  # Transparent background
    )
    
    if text_image:
        # Convert the PIL image to a NumPy array
        text_image_np = np.array(text_image)
        
        # Convert NumPy array to MoviePy ImageClip
        text_clip = ImageClip(text_image_np).set_duration(clip_duration).set_position(("center", "bottom"))

        # Composite the text onto the subclip
        final_subclip = CompositeVideoClip([subclip, text_clip])

        # Add the final subclip to the list of clips
        clips.append(final_subclip)

# Combine all the clips into one video
final_clip = concatenate_videoclips(clips)

# Export the final video
final_clip.write_videofile("output_video_with_countdown.mp4", codec="libx264", fps=24)

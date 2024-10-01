from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, VideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Function to create text image with a specific font
def create_text_image(text, font_path, font_size, text_color, size, bg_color):
    img = Image.new('RGB', size, bg_color)  # Create a background image with RGB
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)  # Load custom font
    except OSError:
        # Fall back to a default PIL font if custom font fails
        font = ImageFont.load_default()
    w, h = draw.textsize(text, font=font)
    position = ((size[0] - w) // 2, (size[1] - h) // 2)  # Center the text
    draw.text(position, text, fill=text_color, font=font)
    return img

# Function to create a countdown frame with PIL
def create_countdown_frame(t, video_duration, font_path, font_size, text_color, size):
    img = Image.new('RGB', size, (0, 0, 0))  # Black background
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    # Calculate remaining time
    remaining_time = max(0, int(video_duration - t))  # Ensure no negative values
    countdown_text = f"{remaining_time:02d}"  # Format text

    # Calculate the position for centering the countdown
    w, h = draw.textsize(countdown_text, font=font)
    position = ((size[0] - w) // 2, (size[1] - h) // 2)

    # Draw the countdown text
    draw.text(position, countdown_text, font=font, fill=text_color)

    # Convert the RGB image to a NumPy array
    return np.array(img)

# Load video
video_path = "output.mp4"
video = VideoFileClip(video_path)

# Font path configuration
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Create text image with custom font
text_image = create_text_image(
    text="IQ Score",
    font_path=font_path,
    font_size=60, 
    text_color=(255, 255, 255),  # White text
    size=(500, 200),  # Width x Height of the text box
    bg_color=(0, 0, 0)  # Black background
)

if text_image:
    text_image_np = np.array(text_image)

    # Convert NumPy array to MoviePy ImageClip for the main text
    text_clip = ImageClip(text_image_np).set_duration(video.duration).set_position(("center", "top"))

    # Create countdown clip using VideoClip and PIL for dynamic countdown rendering
    countdown_clip = VideoClip(
        lambda t: create_countdown_frame(t, video.duration, font_path, 70, (255, 255, 255), (800, 100)),
        duration=video.duration
    ).set_position(("center", "bottom"))

    # Composite the text and countdown onto the video
    final_clip = CompositeVideoClip([video, text_clip, countdown_clip])

    # Export the final video
    final_clip.write_videofile("output_video_with_text_and_countdown.mp4", codec="libx264", fps=24)
else:
    print("Text image creation failed due to font loading issue.")

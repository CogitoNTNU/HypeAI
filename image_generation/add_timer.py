import cv2

# Parameters for the video
input_video = 'output.mp4'  # Replace with your input video file
output_video = 'output_with_timer.mp4'
fps = 30  # Frames per second of the output video
countdown_from = 4  # Start the countdown from this number

# Open the input video file
cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / input_fps

# Get the video dimensions and FPS from the input video

# Create the video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, int(input_fps), (width, height))

# Calculate the number of frames for each second in the countdown
frames_per_second = int(input_fps)

# Initialize the countdown
seconds_remaining = countdown_from

# Function to overlay the timer on the frame
def overlay_timer(frame, seconds):
    # Define the timer text and its properties
    text = f"{int(seconds)}"
    position = (frame.shape[1] - 100, 50)  # Position in the top right corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (0, 0, 255)  # White color
    thickness = 3
    
    # Add the timer text to the frame
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    return frame

# Process each frame of the input video
frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate the seconds remaining in the countdown
    seconds_remaining = countdown_from - (frame_index / frames_per_second)
    
    # Overlay the countdown timer on the frame if there is time remaining
    if seconds_remaining > 0:
        frame = overlay_timer(frame, seconds_remaining)
    
    # Write the modified frame to the output video
    out.write(frame)

    # Move to the next frame
    frame_index += 1

# Release the video capture and writer objects
cap.release()
out.release()

print("Countdown timer added to video and saved as", output_video)

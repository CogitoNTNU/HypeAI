from moviepy.editor import VideoFileClip, AudioFileClip

# Load the video clip
video = VideoFileClip("output_video_with_countdown.mp4")

# Load the audio clip
audio = AudioFileClip("audio.mp3")

# Set the audio of the video to the audio clip
final_video = video.set_audio(audio)

# Export the final video with audio
final_video.write_videofile("output_video_with_audio.mp4", codec="libx264", audio_codec="aac")

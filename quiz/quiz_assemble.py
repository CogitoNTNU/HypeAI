import json
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import os

def create_quiz_video(json_file, video_file, output_dir):
    """Creates a quiz video by overlaying questions on top of the original video at specific timestamps, with timestamps at the end

    Args:
        json_file (_type_): The JSON file containing the quiz questions and answers
        video_file (_type_): The video file to be used as the base for the quiz video
        output_dir (_type_): The output directory where the quiz video will be saved
    """
    
    # Load the JSON file
    with open(json_file, 'r') as file:
        quiz_data = json.load(file)

    # Load the video
    video = VideoFileClip(video_file)
    video_duration = video.duration

    # Calculate segment duration (for 6 segments)
    segment_duration = video_duration / 6

    # Collect questions and answers
    questions = []
    answers = []

    for item in quiz_data:
        if item["type"] == "qa":
            questions.append(item["question"])
            answers.append(item["answer"])

    # Define timestamps for questions
    timestamps = [i * segment_duration for i in range(6)]

    # Create text clips for each question at the respective timestamp
    video_clips = [video]  # Starting with the original video as base

    for i, question in enumerate(questions):
        question_clip = TextClip(question, fontsize=40, color='white', bg_color='black', size=video.size)
        question_clip = question_clip.set_position('center').set_duration(segment_duration).set_start(timestamps[i])
        video_clips.append(question_clip)

    # Generate answer clip to show at the end
    answer_text = "Answers:\n" + "\n".join(answers)
    answer_clip = TextClip(answer_text, fontsize=40, color='white', bg_color='black', size=video.size)
    answer_clip = answer_clip.set_position('center').set_duration(10).set_start(video_duration - 10)  # Last 10 seconds for answers
    video_clips.append(answer_clip)

    # Assemble the video
    final_video = CompositeVideoClip(video_clips)

    # Output the video to the user-specified directory
    output_path = os.path.join(output_dir, "quiz_video.mp4")
    final_video.write_videofile(output_path, codec='libx264')

    print(f"Video successfully saved at: {output_path}")
    
if __name__ == "__main__":
    video_file = 'quiz_output/video_1/107635-678971100_large.mp4'
    create_quiz_video('quiz_prompts/quiz_contents.json', video_file, 'quiz_output/finished')

# Example usage:
# create_quiz_video('path_to_json.json', 'path_to_video.mp4', 'path_to_output_directory')
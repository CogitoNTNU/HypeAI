import json
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
import os

def create_quiz_video(json_file, video_file, audio_file, output_dir):
    """Creates a quiz video by overlaying questions and answers on top of the original video in sequence

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
    
    audio = AudioFileClip(audio_file).subclip(0, video_duration)

    # Collect questions and answers
    questions = []
    answers = []

    for item in quiz_data:
        if item["type"] == "qa":
            questions.append(item["question"])
            answers.append(item["answer"])

    # Determine number of question-answer pairs
    num_pairs = len(questions)

    # Calculate the duration for each pair
    total_duration_per_pair = video_duration / num_pairs
    question_duration = total_duration_per_pair * 0.67  # 67% of the pair duration
    answer_duration = total_duration_per_pair * 0.33    # 33% of the pair duration

    # List to hold all video clips (questions + answers)
    question_answer_clips = []

    # Create text clips for each question and its corresponding answer
    for i, (question, answer) in enumerate(zip(questions, answers)):
        start_time = i * total_duration_per_pair

        # Create the question clip
        question_clip = TextClip(question, fontsize=40, color='white', bg_color='rgba(0, 0, 0, 0)', size=video.size)
        question_clip = question_clip.set_position('center').set_duration(question_duration).set_start(start_time)
        question_answer_clips.append(question_clip)

        # Create the answer clip
        answer_clip = TextClip(answer, fontsize=40, color='yellow', bg_color='rgba(0, 0, 0, 0)', size=video.size)
        answer_clip = answer_clip.set_position('center').set_duration(answer_duration).set_start(start_time + question_duration)
        question_answer_clips.append(answer_clip)

    # Assemble the video
    final_video = CompositeVideoClip([video] + question_answer_clips)
    
    final_video = final_video.set_audio(audio)

    # Output the video to the user-specified directory
    output_path = os.path.join(output_dir, "quiz_video.mp4")
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

    print(f"Video successfully saved at: {output_path}")
    
if __name__ == "__main__":
    video_file = 'quiz_output/video_1/107635-678971100_large.mp4'
    create_quiz_video('quiz_prompts/quiz_contents.json', video_file, 'quiz_output/finished')
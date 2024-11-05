import quiz_generate
from quiz_generate import generate_and_save_all
from quiz_compile_text import generate_to_json
from quiz_assemble import create_quiz_video
import time
import os
import glob

'''
This is the main file for the quiz project. The input is an unprocessed knowledge file
containing information about a topic, and the output is a finished quiz video.
'''

# The original knowledge file
knowledge_file = "knowledge/source_1.txt"

# The output file for the quiz content
quiz_contents = "quiz_prompts/quiz_contents.json"

# The path to the folder where the BG video and music track will be saved
downloaded_components_path = "quiz_output/"

# The output path for the finalised video
output_path = "quiz_output/finished"

def generate_full():
    
    generate_to_json(knowledge_file, quiz_contents)
    generate_and_save_all(quiz_contents, downloaded_components_path)
    
    # Try to find the video file. NB: assuming there is only one video file in the folder
    video_files = glob.glob(os.path.join(downloaded_components_path, "*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video file found in {downloaded_component_path}")
    
    video_file = video_files[0]
    
    create_quiz_video(quiz_contents, video_file, downloaded_components_path + "music_track.mp3", output_path)

if __name__ == "__main__":
    generate_full()
    
    
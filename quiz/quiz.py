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

# The path to the folder where the BG video will be saved
bg_video_path = "quiz_output/video_1"

# The output path for the finalised video
output_path = "quiz_output/finished"

def generate_full():
    
    generate_to_json(knowledge_file, quiz_contents)
    generate_and_save_all(quiz_contents, bg_video_path)
    #time.sleep(5)
    
    # Try to find the video file. NB: assuming there is only one video file in the folder
    video_files = glob.glob(os.path.join(bg_video_path, "*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video file found in {bg_video_path}")
    
    video_file = video_files[0]
    
    create_quiz_video(quiz_contents, video_file, output_path)

if __name__ == "__main__":
    generate_full()
    
    
from quiz_compile_text import generate_to_json
import sys
import os 
import json

'''
PURPOSE:
To download a Pixabay video and generate a music track using Suno API
based on the existing content of the JSON-file with quiz data.
'''

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath('../suno_api_call'))
sys.path.append(os.path.abspath('../pixabay'))

from pixabay_api import get_pixabay_video
from suno_api import generate_and_download

'''
Generate and save the required content for the quiz video.
'''
def generate_and_save_all(input_file, output_directory): 
    
    # Download the pixabay video to the output directory
    get_pixabay_video(get_keyword(input_file), output_directory)
    
    # Generate music track
    #generate_and_download(get_music_prompt(input_file), f"{output_directory}/music_track.mp3")

'''
Get the keyword from the json file. 
'''
def get_keyword(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    for item in data:
        if item['type'] == 'keyword':
            return item['content']
    return None

'''
Get the music prompt froom the json file. 
'''
def get_music_prompt(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    for item in data:
        if item['type'] == 'music_prompt':
            return item['content']
    return None

    
if __name__ == "__main__":
    
    #video_name = "quiz_video"
    
    json_file = "quiz_prompts/quiz_contents.json"
    output_path = "quiz_output/video_1"
    generate_and_save_all(json_file, output_path)
    
    
    #generate_and_save_all()
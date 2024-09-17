from suno_api import generate_audio_by_prompt
from suno_api import download_audio
from suno_api import get_audio_information
from suno_api import download_audio
from suno_api import generate_and_download
import time
import os 
""" 
# Call the generate_audio_by_prompt function with a prompt
prompt = {
  "prompt": "An intense Heavy metal-opera crossover song.",
  "make_instrumental": False,
  "wait_audio": False
}

result = generate_audio_by_prompt(prompt)
print(result)

 # Save the generated audio to a file
#output_file = os.path.join(os.path.dirname(__file__), "audio_output/output_2.wav")
output_file = "/Users/havardvd/Library/CloudStorage/OneDrive-Vikenfylkeskommune/Cogito/Projects/HypeAI/HypeAI/audio_output/output_2.wav"


if result and isinstance(result, list) and 'id' in result[0]:
    # Extract the ID(s)
    audio_id = result[0]['id']
    
    # Step 3: Wait and poll the API to get the audio information
    # You can loop with a timeout or just sleep for a fixed period like 40s
    audio_data = get_audio_information(audio_id)
    audio_url = result[0]['audio_url']
    
    # Wait for the audio to be ready
    while audio_url == '' or audio_url is None:
      time.sleep(30)
      audio_data = get_audio_information(audio_id)
      audio_url = audio_data[0]['audio_url']
    
    audio_data = get_audio_information(audio_id)
    
    # Temporary print statement
    print("Audio data after fetching information:", audio_data)

    # Step 4: If the audio is ready, download to the output file
    download_audio(audio_data, output_file)

else:
    print("Error: 'id' not found in result.")
    
    """

output_file = os.path.join(os.path.dirname(__file__), "audio_output/output_2.wav")
prompt = {
  "prompt": "An intense folk song-blast beat crossover.",
  "make_instrumental": False,
  "wait_audio": False
}
generate_and_download(prompt, output_file)
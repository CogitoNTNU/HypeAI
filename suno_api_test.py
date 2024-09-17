from suno_api import generate_audio_by_prompt
from suno_api import download_audio
from suno_api import get_audio_information
from suno_api import download_audio
import time

# Call the generate_audio_by_prompt function with a prompt
prompt = {
  "prompt": "A soothing song about mice.",
  "make_instrumental": False,
  "wait_audio": False
}

result = generate_audio_by_prompt(prompt)
print(result)

 # Save the generated audio to a file
output_file = "/Users/havardvd/Library/CloudStorage/OneDrive-Vikenfylkeskommune/Cogito/Projects/HypeAI/HypeAI/audio_output/output.wav"

    
if result and isinstance(result, list) and 'id' in result[0]:
    # Extract the ID(s)
    audio_id = result[0]['id']
    
    # Step 3: Wait and poll the API to get the audio information
    # You can loop with a timeout or just sleep for a fixed period like 40s
    time.sleep(120)

    # Fetch audio information by ID
    audio_data = get_audio_information(audio_id)
    print("Audio data after fetching information:", audio_data)

    # Step 4: If the audio is ready, download it
    download_audio(audio_data, output_file)

else:
    print("Error: 'id' not found in result.")

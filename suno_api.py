import time
import requests

# replace your vercel domain

# NB! Remember to change the base_url to the correct URL depending on whether the terminal tells you another port is being used
base_url = 'http://localhost:3001'


def custom_generate_audio(payload):
    url = f"{base_url}/api/custom_generate"
    response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
    return response.json()

def extend_audio(payload):
    url = f"{base_url}/api/extend_audio"
    response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
    return response.json()

def generate_audio_by_prompt(payload):
    url = f"{base_url}/api/generate"
    response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
    return response.json()


def get_audio_information(audio_ids):
    url = f"{base_url}/api/get?ids={audio_ids}"
    response = requests.get(url)
    return response.json()

def get_quota_information():
    url = f"{base_url}/api/get_limit"
    response = requests.get(url)
    return response.json()

def get_clip(clip_id):
    url = f"{base_url}/api/clip?id={clip_id}"
    response = requests.get(url)
    return response.json()

def generate_whole_song(clip_id):
    payloyd = {"clip_id": clip_id}
    url = f"{base_url}/api/concat"
    response = requests.post(url, json=payload)
    return response.json()

# Download audio
def download_audio(data, filename):
    # Check if data is not empty and contains the required keys
    if data and 'audio_url' in data[0]:  # Directly access 'audio_url' in data[0]
        audio_url = data[0]['audio_url']
        if audio_url:  # Check if audio_url is not empty
            response = requests.get(audio_url)
            if response.status_code == 200:  # Ensure the request was successful
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"Audio downloaded successfully as {filename}")
            else:
                print(f"Failed to download audio. Status code: {response.status_code}")
        else:
            print("Audio URL is empty")
    else:
        print("Data does not contain an audio URL")

# Generate and download audio. Requires a full path including file name and extension, and a prompt in this format:
"""
prompt = {
"prompt": "Prompt text.",
"make_instrumental": False,
"wait_audio": False
}   
"""
def generate_and_download(prompt, path):
    result = generate_audio_by_prompt(prompt)
    output_file = path
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


if __name__ == '__main__':
    data = generate_audio_by_prompt({
        "prompt": "A popular heavy metal song about war, sung by a deep-voiced male singer, slowly and melodiously. The lyrics depict the sorrow of people after the war.",
        "make_instrumental": False,
        "wait_audio": False
    })

    ids = f"{data[0]['id']},{data[1]['id']}"
    print(f"ids: {ids}")

    for _ in range(60):
        data = get_audio_information(ids)
        if data[0]["status"] == 'streaming':
            print(f"{data[0]['id']} ==> {data[0]['audio_url']}")
            print(f"{data[1]['id']} ==> {data[1]['audio_url']}")
            break
        # sleep 5s
        time.sleep(5)

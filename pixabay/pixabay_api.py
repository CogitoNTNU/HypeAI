import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Your Pixabay API key
#API_KEY = '38661004-e49591e433508369a5f90fd52' # Move this later
API_KEY = os.getenv('PIXABAY_API_KEY')

'''
Get a video URL and download it from pixabay all in one go. 
'''
def get_pixabay_video(query, output_directory):
    video_url = search_pixabay_videos(query)
    if video_url:
        download_video(video_url, output_directory)
    else:
        print('No video found for this query.')


def search_pixabay_videos(query):
    # Pixabay API endpoint for video search
    url = f'https://pixabay.com/api/videos/?key={API_KEY}&q={query}&per_page=3'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Check if there are videos in the response
        if data['hits']:
            for video in data['hits']:
                if video['duration'] >= 20:
                    video_url = video['videos']['large']['url']
                    return video_url    
            # Return the URL of the first video (HD quality if available)
        else:
            print('No videos found for this query.')
            return None
    else:
        print(f'Error: {response.status_code}')
        return None

def download_video(video_url, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    video_response = requests.get(video_url, stream=True)
    if video_response.status_code == 200:
        # Extract the filename from the URL
        video_filename = os.path.join(output_directory, video_url.split('/')[-1])
        
        # Write video content to a file
        with open(video_filename, 'wb') as video_file:
            for chunk in video_response.iter_content(chunk_size=1024):
                if chunk:
                    video_file.write(chunk)
                    
        print(f'Video downloaded successfully: {video_filename}')
    else:
        print(f'Failed to download video. Status code: {video_response.status_code}')

if __name__ == "__main__":
    # Input for search query and output directory
    search_query = 'Man'
    output_directory = 'quiz/quiz_videos'
    
    get_pixabay_video(search_query, output_directory)

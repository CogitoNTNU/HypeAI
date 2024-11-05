import requests
from bs4 import BeautifulSoup

# Step 1: Send a request to the website
url = 'https://geometry-dash.fandom.com/wiki/Objects#Platforms'  # Replace with the target URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Step 2: Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Step 3: Find the first element with the specified class name
    class_name = 'wds-tab__content wds-is-current'  # Replace with the desired class name
    first_element = soup.find(class_=class_name)
    
    # Step 4: If the element is found, get all <img> tags within it
    if first_element:
        images = first_element.find_all('img')
        
        # Step 5: Extract the 'src' attributes of each image
        image_links = [img.get('src') for img in images]
        
        # Print or process the image URLs
        print("Image URLs within the first element with class '{}':".format(class_name))
        for link in image_links:
            if "gif" not in link:
                print(link)
    else:
        print("No element found with class name '{}'.".format(class_name))
else:
    print("Failed to retrieve the webpage. Status code:", response.status_code)



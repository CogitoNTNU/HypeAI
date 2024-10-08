image_url = "https://hips.hearstapps.com/hmg-prod/images/kanye-west-attends-the-christian-dior-show-as-part-of-the-paris-fashion-week-womenswear-fall-winter-2015-2016-on-march-6-2015-in-paris-france-photo-by-dominique-charriau-wireimage-square.jpg"  # Replace with your image URL
font_path = "/System/Library/Fonts/SFNSRounded.ttf" #for mac
video_text="IQ score"
#Du må ha lastet ned en lydfil med navn "audio.mp3" fil for at add_AIVoice skal funke

from genetic_algorithm import create_basic_video
from test_countdown import add_countdown
from test_bildetekst import add_text
from video_audio_merge import add_AIVoice

create_basic_video(image_url)
add_countdown()
add_text(font_path,video_text)
add_AIVoice()

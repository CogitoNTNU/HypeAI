from suno_api import generate_audio_by_prompt
from suno_api import custom_generate_audio
from suno_api import download_audio
from suno_api import get_audio_information
from suno_api import download_audio
from suno_api import generate_and_download
import time
import os 
import os

# Get the root directory of the project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Construct the path relative to the root directory
output_file = os.path.join(root_dir, "audio_output/output_2.wav")

# output_file = os.path.join(os.path.dirname(__file__), "audio_output/suno/output_2.wav")
prompt = {
  "prompt": "A happy-go-lucky folk song.",
  "make_instrumental": False,
  "payload": "A happy-go-lucky folk song."
}
generate_and_download(prompt, output_file)
#generate_audio_by_prompt(prompt)
#custom_generate_audio(prompt)

# SUNO_COOKIE=__cf_bm=dZw4mf9CQ3mz37Oj5fx0OkcTw_3y4USU0aQi4YNtzyY-1725383795-1.0.1.1-nxPQzQK6LuziQdxL2CDJkhV3kx4KQMgBiJeypDqLKNj1XCD69btaNE3bA8BSeDLB7XAFU19RmtVDIJhdteTKxA; _cfuvid=4.3_6nOA6r.AMQrxhMnjkvrg_Q2jxQyU5K_otlglhjI-1725383795719-0.0.1.1-604800000; ajs_anonymous_id=c0b666bf-959a-4412-a4b6-40d438190f5a; __client=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImNsaWVudF8ybFpObzRoNnNjRUE5QXlLZWNTSVA3aVNPa0QiLCJyb3RhdGluZ190b2tlbiI6Imp6Z3VkaXRjc2xsOGp4bXE1Yms4ZTg5dDBmYXRva3V5N2UzZzg0amEifQ.KdNPRx4a_e756YVFZWQzFyPGvBdzFVZkouFuZFGpelTsc2tYPefZO0KjE0F92O_FeGnFSiAPptHY6GHxRQfkTHACGFjHkm_FfYF0jmJ_TU4lXShA5H6_oHFyHWKG9gHk_zDG4FhxmBwYc8WRMpWowqaqR132l_XKElVj5krRrft79njXaN-YgPohqBwnw3AfPdqoaL4zTSJ9GL90gJkA-I6dvKcBoGtItt68OgMT4XLha6DMc3jqT29dwIHOfBU2OKl27BcjDb04ly9aypIu64LINkQJQ5kcJ8Wv8zrl_yGVvLKxKDfutOa2L6ZXZaF5HxL_bjVKxL17ZeoDHUxM1w; __client_uat=1725383814; __stripe_mid=e0b2710d-77b3-43dd-988d-a36af911d7dd65740d; __stripe_sid=cb6679b8-b449-4687-b8d3-39899ba5160ed742b6

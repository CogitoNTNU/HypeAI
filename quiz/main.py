import replicate
from dotenv import load_dotenv

load_dotenv()

output = replicate.run(
    "black-forest-labs/flux-pro",
    input={
        "steps": 25,
        "prompt": "Joe Biden robbing a bank with an ak-47 showing his face and wearing a suit",
        "guidance": 3,
        "interval": 2,
        "aspect_ratio": "1:1",
        "output_format": "webp",
        "output_quality": 80,
        "safety_tolerance": 2
    }
)

print(output)
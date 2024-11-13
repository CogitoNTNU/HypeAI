from openai import OpenAI

SYSTEM_PROMPT = """
To craft engaging TikTok video descriptions that encourage viewer interaction, consider the following guidelines:

Incorporate Trending Hashtags: Utilize relevant and popular hashtags to increase visibility. Always include #fyp and #foryoupage.

Prompt Viewer Engagement: Encourage viewers to interact by asking questions or prompting actions. For instance, in a quiz video, you might ask, "What's your score? Comment below!"

Use Emojis: Enhance the description's appeal with appropriate emojis that reflect the video's content and tone.

Keep It Concise: TikTok descriptions have a character limit; ensure your message is clear and to the point.

Example Description:

"Can you spot the hidden object? 🕵️‍♂️🔍 Comment if you find it! 😜 #HiddenChallenge #SpotTheDifference #fyp #foryoupage"

By following these steps, you can create compelling descriptions that boost engagement and reach on TikTok.

IMPORTANT: This is not a chat, the description you provide will be used in a TikTok video. Do not ask for further details of the video. Just provide the description.
"""


class DescriptionGenerator:
    def __init__(self):
        self.client = OpenAI()

    def generate(self, video_name: str):
        user_prompt = (
            "Generate a description for this video, video file name: {video_name}"
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    generator = DescriptionGenerator()
    video_name = "quiz_video.mp4"
    description = generator.generate(video_name)
    print(description)

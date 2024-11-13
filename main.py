from dotenv import load_dotenv
from automator.video_automator import VideoAutomator
from automator.video_uploader import TikTokUploader
from automator.video_uploader import MockUploader
from automator.video_generator import MockGenerator


def main():
    print("Starting the TikTok video automator")
    load_dotenv()
    automator = VideoAutomator()

    uploader = TikTokUploader("./automator/cookies.txt")
    automator.add_video_uploader(uploader)

    if automator.has_videos():
        print("Videos already generated, uploading a random video")
        automator.upload_random_video()
        return

    automator.add_video_generator(MockGenerator())

    automator.generate_videos()
    automator.upload_random_video()


if __name__ == "__main__":
    main()

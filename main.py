from automator.video_automator import VideoAutomator
from automator.video_uploader import TikTokUploader
from automator.video_uploader import MockUploader
from automator.video_generator import MockGenerator


def main():
    automator = VideoAutomator()

    if automator.has_videos():
        automator.upload_random_video()
        return

    uploader = TikTokUploader("./automator/cookies.txt")
    automator.add_video_uploader(uploader)

    automator.add_video_uploader(MockUploader())
    automator.add_video_generator(MockGenerator())

    automator.generate_videos()
    automator.upload_random_video()

if __name__ == "__main__":
    main()

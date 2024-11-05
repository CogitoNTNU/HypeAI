from video_uploader import TikTokUploader
from video_uploader import MockUploader

from video_automator import VideoAutomator
from video_generator import MockGenerator


def main():
    automator = VideoAutomator()

    # uploader = TikTokUploader("cookies.txt")
    # automator.add_video_uploader(uploader)

    automator.add_video_uploader(MockUploader())
    automator.add_video_generator(MockGenerator())

    automator.generate_videos()
    automator.upload_videos()

    print("Clearing output folder...")
    # automator.clear_output_folder()


if __name__ == "__main__":
    main()

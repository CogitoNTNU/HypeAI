from video_uploader import TikTokUploader
from video_automator import VideoAutomator


def main():
    # DOES NOT WORK
    uploader = TikTokUploader("cookies.txt")

    automator = VideoAutomator()
    automator.add_video_uploader(uploader)

    # uploader.upload("./output/video.mp4")


if __name__ == "__main__":
    main()

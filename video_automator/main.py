from video_uploader import TikTokUploader


def main():
    # DOES NOT WORK
    uploader = TikTokUploader("cookies.txt")
    uploader.upload("./output/video.mp4")


if __name__ == "__main__":
    main()

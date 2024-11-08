from abc import ABC, abstractmethod
import os
import random
from automator.video_uploader import VideoUploader
from automator.video_generator import VideoGenerator

OUTPUT_PATH = "automator/output/"


class VideoAutomator:
    def __init__(self) -> None:
        self.video_generators: list[VideoGenerator] = []
        self.video_uploaders: list[VideoUploader] = []

    def add_video_generator(self, video_generator: VideoGenerator) -> None:
        self.video_generators.append(video_generator)

    def add_video_uploader(self, video_uploader: VideoUploader) -> None:
        self.video_uploaders.append(video_uploader)

    def generate_videos(self) -> None:
        for video_generator in self.video_generators:
            video_generator.generate(output_path=OUTPUT_PATH)

    def upload_random_video(self) -> None:
        videos = os.listdir(OUTPUT_PATH)
        if not videos:
            print("No videos to upload")
            return

        video = random.choice(videos)
        for video_uploader in self.video_uploaders:
            video_uploader.upload(video_path=os.path.join(OUTPUT_PATH, video))
            self.remove_video(os.path.join(OUTPUT_PATH, video))

    def has_videos(self) -> bool:
        return bool(os.listdir(OUTPUT_PATH))

    def remove_video(self, video_path: str) -> None:
        os.remove(video_path)

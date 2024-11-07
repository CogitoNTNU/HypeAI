from abc import ABC, abstractmethod
import os
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

    def upload_videos(self) -> None:
        for video in os.listdir(OUTPUT_PATH):
            for video_uploader in self.video_uploaders:
                video_uploader.upload(video_path=os.path.join(OUTPUT_PATH, video))

    def clear_output_folder(self) -> None:
        for video in os.listdir(OUTPUT_PATH):
            os.remove(os.path.join(OUTPUT_PATH, video))

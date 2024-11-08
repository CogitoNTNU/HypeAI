from typing import Protocol


class VideoGenerator(Protocol):
    def generate(self, output_path: str) -> None:
        """
        Generates a video and returns the path to the generated video
        """
        ...

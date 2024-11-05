from abc import ABC, abstractmethod


class VideoGenerator(ABC):
    @abstractmethod
    def generate(self, output_path: str) -> None:
        """
        Generates a video and returns the path to the generated video
        """
        pass

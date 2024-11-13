from abc import ABC, abstractmethod


class VideoUploader(ABC):
    @abstractmethod
    def upload(self, video_path: str, description="") -> None:
        pass

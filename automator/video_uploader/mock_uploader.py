from .video_uploader import VideoUploader


class MockUploader(VideoUploader):
    def __init__(self) -> None:
        pass

    def upload(self, video_path: str) -> None:
        print(f"Uploading {video_path} (mock)...")
        print("Upload complete")

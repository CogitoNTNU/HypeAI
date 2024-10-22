import tiktok_uploader.upload as tiktok
from tiktok_uploader.auth import AuthBackend
from .video_uploader import VideoUploader


class TikTokUploader(VideoUploader):
    def __init__(self, cookies_path: str) -> None:
        self.cookies_path = cookies_path
        self.auth = AuthBackend(cookies=cookies_path)

    def upload(self, video_path: str) -> None:
        tiktok.upload_videos([video_path], self.auth)

    def upload_multiple(self, video_paths: list[str]) -> None:
        tiktok.upload_videos(video_paths, self.auth)

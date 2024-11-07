from .video_generator import VideoGenerator


class MockGenerator(VideoGenerator):
    def generate(self, output_path: str) -> None:
        print(f"Generating video at {output_path}...")
        print(f"Video generated at {output_path}")

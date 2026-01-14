from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from pathlib import Path
from src.core.video_processor import VideoProcessor

class FrameExtractionInput(BaseModel):
    video_path: str = Field(description="Path to the video file")
    output_dir: str = Field(description="Directory to save extracted frames")
    max_frames: Optional[int] = Field(default=None, description="Maximum number of frames to extract")
    sample_rate: Optional[int] = Field(default=None, description="Frame sampling rate")

class FrameExtractionTool(BaseTool):
    name = "extract_frames"
    description = "Extract frames from a video file"
    args_schema: Type[BaseModel] = FrameExtractionInput
    
    def _run(self, video_path: str, output_dir: str, max_frames: Optional[int] = None, sample_rate: Optional[int] = None) -> str:
        processor = VideoProcessor()
        frames = processor.extract_frames(
            Path(video_path),
            Path(output_dir),
            max_frames=max_frames,
            sample_rate=sample_rate
        )
        return f"Extracted {len(frames)} frames to {output_dir}"

class KeyframeDetectionInput(BaseModel):
    video_path: str = Field(description="Path to the video file")
    output_dir: str = Field(description="Directory to save keyframes")
    threshold: float = Field(default=30.0, description="Scene change detection threshold")
    max_frames: Optional[int] = Field(default=None, description="Maximum number of keyframes")
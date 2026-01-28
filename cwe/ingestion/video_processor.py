"""Video processing for frame extraction and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from cwe.models.artifact import VideoArtifact, VideoFrame


@dataclass
class VideoProcessorConfig:
    """Configuration for video processing."""
    
    # Frame extraction
    keyframe_interval: float = 1.0  # Extract keyframe every N seconds
    max_keyframes: int = 100
    
    # Scene detection
    enable_scene_detection: bool = True
    scene_threshold: float = 30.0
    
    # Output
    output_format: str = "jpg"
    output_quality: int = 85


class VideoProcessor:
    """
    Processes video artifacts for analysis.
    
    Capabilities:
    - Frame extraction at configurable intervals
    - Keyframe/scene change detection
    - OCR for embedded timestamps
    - Basic motion/object detection (future)
    """
    
    def __init__(self, config: VideoProcessorConfig | None = None):
        self.config = config or VideoProcessorConfig()
    
    def process(self, video_path: Path | str, output_dir: Path | str) -> VideoArtifact:
        """
        Process a video file and extract frames.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            
        Returns:
            VideoArtifact with metadata and frame references
        """
        import cv2
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Create artifact
        artifact = VideoArtifact(
            incident_id=None,  # Set by caller
            filename=video_path.name,
            file_path=str(video_path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=timedelta(seconds=duration),
            extracted_frames_dir=str(output_dir),
        )
        
        # Extract keyframes
        keyframes = list(self._extract_keyframes(cap, output_dir, fps))
        artifact.keyframes = keyframes
        
        cap.release()
        
        return artifact
    
    def _extract_keyframes(
        self,
        cap,
        output_dir: Path,
        fps: float,
    ) -> Iterator[VideoFrame]:
        """Extract keyframes from video."""
        import cv2
        
        frame_interval = int(fps * self.config.keyframe_interval)
        frame_number = 0
        extracted = 0
        prev_frame = None
        
        while extracted < self.config.max_keyframes:
            ret, frame = cap.read()
            if not ret:
                break
            
            is_keyframe = False
            
            # Extract at regular intervals
            if frame_number % frame_interval == 0:
                is_keyframe = True
            
            # Scene detection
            if self.config.enable_scene_detection and prev_frame is not None:
                diff = cv2.absdiff(frame, prev_frame)
                score = diff.mean()
                if score > self.config.scene_threshold:
                    is_keyframe = True
            
            if is_keyframe:
                # Save frame
                frame_path = output_dir / f"frame_{frame_number:06d}.{self.config.output_format}"
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.output_quality])
                
                timestamp_offset = frame_number / fps
                
                yield VideoFrame(
                    frame_number=frame_number,
                    timestamp=datetime.utcnow(),  # Placeholder - would need video timestamp
                    timestamp_offset=timestamp_offset,
                    file_path=str(frame_path),
                    is_keyframe=True,
                )
                
                extracted += 1
            
            prev_frame = frame.copy()
            frame_number += 1
    
    def extract_ocr_timestamps(self, artifact: VideoArtifact) -> list[dict]:
        """
        Extract timestamps from video frames using OCR.
        
        Useful for aligning video with external time sources.
        """
        # TODO: Implement OCR timestamp extraction
        # Would use pytesseract or similar
        return []

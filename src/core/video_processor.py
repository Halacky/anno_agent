import cv2
from pathlib import Path
from typing import List, Optional
import numpy as np
from loguru import logger

class VideoProcessor:
    @staticmethod
    def extract_frames(
        video_path: Path,
        output_dir: Path,
        max_frames: Optional[int] = None,
        sample_rate: Optional[int] = None
    ) -> List[Path]:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video info - Total frames: {total_frames}, FPS: {fps}")
        
        if max_frames and total_frames > max_frames:
            sample_rate = total_frames // max_frames
        elif sample_rate is None:
            sample_rate = 1
        
        extracted_frames = []
        frame_idx = 0
        saved_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                frame_path = output_dir / f"frame_{saved_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_frames.append(frame_path)
                saved_idx += 1
                
                if max_frames and saved_idx >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Extracted {len(extracted_frames)} frames from video")
        
        return extracted_frames
    
    @staticmethod
    def detect_scene_changes(
        video_path: Path,
        output_dir: Path,
        threshold: float = 30.0,
        max_frames: Optional[int] = None
    ) -> List[Path]:
        cap = cv2.VideoCapture(str(video_path))
        
        prev_frame = None
        keyframes = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                
                if mean_diff > threshold:
                    frame_path = output_dir / f"keyframe_{len(keyframes):06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    keyframes.append(frame_path)
                    
                    if max_frames and len(keyframes) >= max_frames:
                        break
            
            prev_frame = gray
            frame_idx += 1
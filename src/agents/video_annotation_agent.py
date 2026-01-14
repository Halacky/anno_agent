from pathlib import Path
from typing import Dict, List, Optional
import asyncio
from src.agents.base_agent import BaseAnnotationAgent
from src.core.video_processor import VideoProcessor
from src.prompts.templates import (
    ANNOTATION_SYSTEM_PROMPT,
    VIDEO_ANNOTATION_PROMPT,
    VIDEO_WITH_KEYFRAMES_PROMPT
)
from src.utils.file_utils import (
    create_output_structure,
    save_annotation,
    load_annotation
)

class VideoAnnotationAgent(BaseAnnotationAgent):
    async def annotate_video_basic(
        self,
        video_path: Path,
        task_description: str,
        output_dir: Optional[Path] = None,
        max_frames: int = 100
    ) -> Dict:
        """Вариант 1: Базовая разметка видео без опорных кадров"""
        
        self.logger.info(f"Starting basic video annotation for {video_path}")
        
        from config.settings import settings
        output_dir = output_dir or settings.output_dir
        output_structure = create_output_structure(output_dir, "video_basic")
        
        # Извлечение кадров
        processor = VideoProcessor()
        frames = processor.extract_frames(
            video_path,
            output_structure['frames'],
            max_frames=max_frames
        )
        
        # Аннотирование каждого кадра
        annotations = {}
        total_frames = len(frames)
        
        for idx, frame_path in enumerate(frames):
            self.logger.info(f"Annotating frame {idx + 1}/{total_frames}")
            
            prompt = VIDEO_ANNOTATION_PROMPT.format(
                task_description=task_description,
                frame_number=idx + 1,
                total_frames=total_frames
            )
            
            result = await self.qwen_client.analyze_image(
                frame_path,
                prompt,
                system_prompt=ANNOTATION_SYSTEM_PROMPT
            )
            
            if result["success"]:
                annotation_data = await self._parse_annotation_response(result["content"])
                annotation_data["frame_path"] = str(frame_path)
                annotation_data["frame_number"] = idx + 1
                
                annotation_file = output_structure['annotations'] / f"frame_{idx:06d}.json"
                save_annotation(annotation_data, annotation_file)
                
                annotations[frame_path.name] = annotation_data
            else:
                self.logger.error(f"Failed to annotate frame {idx}: {result.get('error')}")
        
        # Сохранение метаданных
        metadata = {
            "video_path": str(video_path),
            "task_description": task_description,
            "total_frames": total_frames,
            "annotated_frames": len(annotations),
            "output_structure": {k: str(v) for k, v in output_structure.items()}
        }
        
        save_annotation(metadata, output_structure['metadata'] / "task_metadata.json")
        
        self.logger.info(f"Video annotation completed. Results saved to {output_structure['root']}")
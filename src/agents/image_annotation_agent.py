from pathlib import Path
from typing import Dict, List, Optional
import asyncio
from src.agents.base_agent import BaseAnnotationAgent
from src.prompts.templates import (
    ANNOTATION_SYSTEM_PROMPT,
    IMAGE_ANNOTATION_PROMPT,
    IMAGE_WITH_EXAMPLES_PROMPT
)
from src.utils.file_utils import (
    create_output_structure,
    save_annotation,
    get_image_files,
    load_annotation
)

class ImageAnnotationAgent(BaseAnnotationAgent):
    async def annotate_images_with_examples(
        self,
        images_dir: Path,
        examples_dir: Path,
        task_description: str,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """Вариант 3: Разметка изображений с примерами"""
        
        self.logger.info(f"Starting image annotation with examples from {examples_dir}")
        
        from config.settings import settings
        output_dir = output_dir or settings.output_dir
        output_structure = create_output_structure(output_dir, "images_with_examples")
        
        # Загрузка примеров разметки
        example_files = get_image_files(examples_dir)
        example_annotations = []
        
        for example_img in example_files[:5]:  # Ограничиваем 5 примерами
            annotation_file = example_img.parent / f"{example_img.stem}_annotation.json"
            
            if annotation_file.exists():
                example_ann = load_annotation(annotation_file)
                example_annotations.append({
                    "image": example_img.name,
                    "annotation": example_ann
                })
            else:
                # Если нет готовой разметки, создаем пример
                self.logger.warning(f"No annotation file for {example_img.name}, analyzing...")
                result = await self.qwen_client.analyze_image(
                    example_img,
                    IMAGE_ANNOTATION_PROMPT.format(
                        task_description=task_description,
                        image_name=example_img.name
                    ),
                    system_prompt=ANNOTATION_SYSTEM_PROMPT
                )
                
                if result["success"]:
                    ann = await self._parse_annotation_response(result["content"])
                    example_annotations.append({
                        "image": example_img.name,
                        "annotation": ann
                    })
        
        # Форматирование примеров для контекста
        examples_context = "\n\n".join([
            f"Example {i+1} - Image: {ex['image']}\nAnnotation: {ex['annotation']}"
            for i, ex in enumerate(example_annotations)
        ])
        
        # Получение целевых изображений
        target_images = get_image_files(images_dir)
        annotations = {}
        
        for idx, img_path in enumerate(target_images):
            self.logger.info(f"Annotating image {idx + 1}/{len(target_images)}: {img_path.name}")
            
            prompt = IMAGE_WITH_EXAMPLES_PROMPT.format(
                task_description=task_description,
                image_name=img_path.name,
                example_annotations=examples_context
            )
            
            result = await self.qwen_client.analyze_image(
                img_path,
                prompt,
                system_prompt=ANNOTATION_SYSTEM_PROMPT
            )
            
            if result["success"]:
                annotation_data = await self._parse_annotation_response(result["content"])
                annotation_data["image_path"] = str(img_path)
                annotation_data["used_examples"] = len(example_annotations)
                
                annotation_file = output_structure['annotations'] / f"{img_path.stem}.json"
                save_annotation(annotation_data, annotation_file)
                
                annotations[img_path.name] = annotation_data
        
        # Метаданные
        metadata = {
            "images_dir": str(images_dir),
            "examples_dir": str(examples_dir),
            "task_description": task_description,
            "total_images": len(target_images),
            "annotated_images": len(annotations),
            "examples_used": len(example_annotations),
            "output_structure": {k: str(v) for k, v in output_structure.items()}
        }
        
        save_annotation(metadata, output_structure['metadata'] / "task_metadata.json")
        
        self.logger.info(f"Image annotation with examples completed. Results saved to {output_structure['root']}")
        
        return {
            "status": "success",
            "metadata": metadata,
            "annotations": annotations,
            "output_dir": str(output_structure['root'])
        }
    
    async def annotate_images_basic(
        self,
        images_dir: Path,
        task_description: str,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """Вариант 4: Базовая разметка изображений без примеров"""
        
        self.logger.info(f"Starting basic image annotation for {images_dir}")
        
        from config.settings import settings
        output_dir = output_dir or settings.output_dir
        output_structure = create_output_structure(output_dir, "images_basic")
        
        # Получение изображений
        target_images = get_image_files(images_dir)
        annotations = {}
        
        for idx, img_path in enumerate(target_images):
            self.logger.info(f"Annotating image {idx + 1}/{len(target_images)}: {img_path.name}")
            
            prompt = IMAGE_ANNOTATION_PROMPT.format(
                task_description=task_description,
                image_name=img_path.name
            )
            
            result = await self.qwen_client.analyze_image(
                img_path,
                prompt,
                system_prompt=ANNOTATION_SYSTEM_PROMPT
            )
            
            if result["success"]:
                annotation_data = await self._parse_annotation_response(result["content"])
                annotation_data["image_path"] = str(img_path)
                
                annotation_file = output_structure['annotations'] / f"{img_path.stem}.json"
                save_annotation(annotation_data, annotation_file)
                
                annotations[img_path.name] = annotation_data
        
        # Метаданные
        metadata = {
            "images_dir": str(images_dir),
            "task_description": task_description,
            "total_images": len(target_images),
            "annotated_images": len(annotations),
            "output_structure": {k: str(v) for k, v in output_structure.items()}
        }
        
        save_annotation(metadata, output_structure['metadata'] / "task_metadata.json")
        
        self.logger.info(f"Basic image annotation completed. Results saved to {output_structure['root']}")
        
        return {
            "status": "success",
            "metadata": metadata,
            "annotations": annotations,
            "output_dir": str(output_structure['root'])
        }
    
    async def annotate_video_with_keyframes(
        self,
        video_path: Path,
        task_description: str,
        keyframe_annotations: Dict[Path, Dict],
        output_dir: Optional[Path] = None,
        max_frames: int = 100
    ) -> Dict:
        """Вариант 2: Разметка видео с использованием опорных кадров"""
        
        self.logger.info(f"Starting keyframe-guided video annotation for {video_path}")
        
        from config.settings import settings
        output_dir = output_dir or settings.output_dir
        output_structure = create_output_structure(output_dir, "video_keyframes")
        
        # Извлечение всех кадров
        processor = VideoProcessor()
        frames = processor.extract_frames(
            video_path,
            output_structure['frames'],
            max_frames=max_frames
        )
        
        # Форматирование примеров опорных кадров
        keyframe_examples = []
        for kf_path, kf_ann in keyframe_annotations.items():
            keyframe_examples.append(f"Keyframe: {kf_path.name}\nAnnotation: {kf_ann}")
        
        keyframe_context = "\n\n".join(keyframe_examples)
        
        # Аннотирование с учетом опорных кадров
        annotations = {}
        total_frames = len(frames)
        
        for idx, frame_path in enumerate(frames):
            self.logger.info(f"Annotating frame {idx + 1}/{total_frames} with keyframe guidance")
            
            prompt = VIDEO_WITH_KEYFRAMES_PROMPT.format(
                task_description=task_description,
                frame_number=idx + 1,
                keyframe_annotations=keyframe_context
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
                annotation_data["used_keyframe_guidance"] = True
                
                annotation_file = output_structure['annotations'] / f"frame_{idx:06d}.json"
                save_annotation(annotation_data, annotation_file)
                
                annotations[frame_path.name] = annotation_data
        
        # Метаданные
        metadata = {
            "video_path": str(video_path),
            "task_description": task_description,
            "total_frames": total_frames,
            "annotated_frames": len(annotations),
            "keyframes_used": len(keyframe_annotations),
            "output_structure": {k: str(v) for k, v in output_structure.items()}
        }
        
        save_annotation(metadata, output_structure['metadata'] / "task_metadata.json")
        
        self.logger.info(f"Keyframe-guided annotation completed. Results saved to {output_structure['root']}")
        
        return {
            "status": "success",
            "metadata": metadata,
            "annotations": annotations,
            "output_dir": str(output_structure['root'])
        }
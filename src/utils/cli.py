
import argparse
import asyncio
from pathlib import Path
from src.agents.video_annotation_agent import VideoAnnotationAgent
from src.agents.image_annotation_agent import ImageAnnotationAgent
from src.utils.logging_config import setup_logging
from src.utils.file_utils import load_annotation
import json

logger = setup_logging()

def create_parser():
    """Создание парсера аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="AI Annotation Agent - Automated video and image annotation"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Команда для видео (базовая)
    video_basic = subparsers.add_parser('video-basic', help='Annotate video (basic)')
    video_basic.add_argument('video_path', type=Path, help='Path to video file')
    video_basic.add_argument('--task', '-t', required=True, help='Task description')
    video_basic.add_argument('--max-frames', type=int, default=100, help='Max frames to extract')
    video_basic.add_argument('--output', '-o', type=Path, help='Output directory')
    
    # Команда для видео с опорными кадрами
    video_keyframes = subparsers.add_parser('video-keyframes', help='Annotate video with keyframes')
    video_keyframes.add_argument('video_path', type=Path, help='Path to video file')
    video_keyframes.add_argument('keyframes_dir', type=Path, help='Directory with keyframe annotations')
    video_keyframes.add_argument('--task', '-t', required=True, help='Task description')
    video_keyframes.add_argument('--max-frames', type=int, default=100, help='Max frames')
    video_keyframes.add_argument('--output', '-o', type=Path, help='Output directory')
    
    # Команда для изображений с примерами
    images_examples = subparsers.add_parser('images-examples', help='Annotate images with examples')
    images_examples.add_argument('images_dir', type=Path, help='Directory with images to annotate')
    images_examples.add_argument('examples_dir', type=Path, help='Directory with example annotations')
    images_examples.add_argument('--task', '-t', required=True, help='Task description')
    images_examples.add_argument('--output', '-o', type=Path, help='Output directory')
    
    # Команда для изображений (базовая)
    images_basic = subparsers.add_parser('images-basic', help='Annotate images (basic)')
    images_basic.add_argument('images_dir', type=Path, help='Directory with images')
    images_basic.add_argument('--task', '-t', required=True, help='Task description')
    images_basic.add_argument('--output', '-o', type=Path, help='Output directory')
    
    return parser

async def main():
    """Главная функция CLI"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'video-basic':
            agent = VideoAnnotationAgent()
            result = await agent.annotate_video_basic(
                video_path=args.video_path,
                task_description=args.task,
                output_dir=args.output,
                max_frames=args.max_frames
            )
            await agent.cleanup()
            
        elif args.command == 'video-keyframes':
            agent = VideoAnnotationAgent()
            
            # Загрузка аннотаций опорных кадров
            keyframe_annotations = {}
            for ann_file in args.keyframes_dir.glob("*_annotation.json"):
                img_name = ann_file.stem.replace("_annotation", "")
                img_path = args.keyframes_dir / f"{img_name}.jpg"
                if not img_path.exists():
                    img_path = args.keyframes_dir / f"{img_name}.png"
                
                if img_path.exists():
                    keyframe_annotations[img_path] = load_annotation(ann_file)
            
            result = await agent.annotate_video_with_keyframes(
                video_path=args.video_path,
                task_description=args.task,
                keyframe_annotations=keyframe_annotations,
                output_dir=args.output,
                max_frames=args.max_frames
            )
            await agent.cleanup()
            
        elif args.command == 'images-examples':
            agent = ImageAnnotationAgent()
            result = await agent.annotate_images_with_examples(
                images_dir=args.images_dir,
                examples_dir=args.examples_dir,
                task_description=args.task,
                output_dir=args.output
            )
            await agent.cleanup()
            
        elif args.command == 'images-basic':
            agent = ImageAnnotationAgent()
            result = await agent.annotate_images_basic(
                images_dir=args.images_dir,
                task_description=args.task,
                output_dir=args.output
            )
            await agent.cleanup()
        
        # Вывод результатов
        logger.info("=" * 60)
        logger.info("ANNOTATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Status: {result['status']}")
        logger.info(f"Output: {result['output_dir']}")
        logger.info(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
"""
Пример 1: Базовая разметка видео без опорных кадров

Использование:
python examples/example_1_video_basic.py
"""

import asyncio
from pathlib import Path
from src.agents.video_annotation_agent import VideoAnnotationAgent
from src.utils.logging_config import setup_logging

# Настройка логирования
logger = setup_logging()

async def main():
    """
    Вариант 1: Базовая разметка видео
    
    Входные данные:
    - Видео файл
    - Описание задачи разметки
    
    Выходные данные:
    - Извлеченные кадры
    - JSON файлы с разметкой для каждого кадра
    """
    
    # Инициализация агента
    agent = VideoAnnotationAgent()
    
    # Параметры задачи
    video_path = Path("data/input/sample_video.mp4")
    task_description = """
    Detect and annotate all persons in the video frames.
    For each person, provide:
    - Bounding box coordinates [x_min, y_min, x_max, y_max]
    - Visibility score (0.0 to 1.0)
    - Pose estimation if visible (standing, sitting, walking, etc.)
    - Any visible attributes (clothing color, accessories)
    """
    
    # Параметры обработки
    max_frames = 50  # Ограничение на количество кадров
    output_dir = Path("output/video_annotations")
    
    try:
        logger.info("Starting video annotation task...")
        
        # Запуск разметки
        result = await agent.annotate_video_basic(
            video_path=video_path,
            task_description=task_description,
            output_dir=output_dir,
            max_frames=max_frames
        )
        
        # Вывод результатов
        logger.info(f"Annotation completed successfully!")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Total frames: {result['metadata']['total_frames']}")
        logger.info(f"Annotated frames: {result['metadata']['annotated_frames']}")
        logger.info(f"Output directory: {result['output_dir']}")
        
        # Пример первой аннотации
        if result['annotations']:
            first_annotation = list(result['annotations'].values())[0]
            logger.info(f"\nExample annotation (first frame):")
            logger.info(f"Frame: {first_annotation['frame_number']}")
            logger.info(f"Objects found: {len(first_annotation.get('annotations', []))}")
            
            for obj in first_annotation.get('annotations', [])[:3]:  # Показываем первые 3
                logger.info(f"  - {obj.get('object')}: confidence {obj.get('confidence', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error during annotation: {e}")
        raise
    
    finally:
        # Очистка ресурсов
        await agent.cleanup()
        logger.info("Agent cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())
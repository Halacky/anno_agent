"""
Пример 2: Разметка видео с использованием опорных кадров

Использование:
python examples/example_2_video_keyframes.py
"""

import asyncio
from pathlib import Path
from src.agents.video_annotation_agent import VideoAnnotationAgent
from src.utils.logging_config import setup_logging
from src.utils.file_utils import load_annotation

# Настройка логирования
logger = setup_logging()

async def prepare_keyframe_annotations(keyframe_dir: Path, agent: VideoAnnotationAgent) -> dict:
    """
    Подготовка аннотаций для опорных кадров
    
    Если аннотации уже существуют - загружаем их,
    иначе создаем новые с помощью агента
    """
    keyframe_annotations = {}
    
    keyframe_files = list(keyframe_dir.glob("*.jpg")) + list(keyframe_dir.glob("*.png"))
    
    for kf_path in keyframe_files:
        annotation_file = kf_path.parent / f"{kf_path.stem}_annotation.json"
        
        if annotation_file.exists():
            # Загружаем существующую аннотацию
            annotation = load_annotation(annotation_file)
            keyframe_annotations[kf_path] = annotation
            logger.info(f"Loaded existing annotation for {kf_path.name}")
        else:
            # Создаем новую аннотацию (если нужно)
            logger.info(f"Creating annotation for keyframe {kf_path.name}")
            # Здесь можно добавить логику создания аннотации
            # Пока просто пропускаем
            pass
    
    return keyframe_annotations

async def main():
    """
    Вариант 2: Разметка видео с опорными кадрами
    
    Входные данные:
    - Видео файл
    - Описание задачи разметки
    - Предварительная разметка для опорных кадров
    
    Выходные данные:
    - Извлеченные кадры
    - JSON файлы с разметкой для каждого кадра
    """
    
    # Инициализация агента
    agent = VideoAnnotationAgent()
    
    # Параметры задачи
    video_path = Path("data/input/sample_video.mp4")
    keyframes_dir = Path("data/keyframes")  # Папка с опорными кадрами
    
    task_description = """
    Detect and track vehicles across video frames.
    For each vehicle, provide:
    - Bounding box coordinates [x_min, y_min, x_max, y_max]
    - Vehicle type (car, truck, bus, motorcycle, etc.)
    - Color if visible
    - Direction of movement (left, right, towards, away)
    - Tracking ID to maintain consistency across frames
    
    Use the keyframe annotations as reference to maintain consistent tracking IDs
    and annotation style throughout the video.
    """
    
    # Параметры обработки
    max_frames = 100
    output_dir = Path("output/video_keyframes_annotations")
    
    try:
        logger.info("Preparing keyframe annotations...")
        
        # Подготовка аннотаций опорных кадров
        # В реальном сценарии эти аннотации создаются вручную или другим способом
        keyframe_annotations = {
            keyframes_dir / "keyframe_0001.jpg": {
                "annotations": [
                    {
                        "object": "car",
                        "bbox": [100, 150, 250, 300],
                        "confidence": 0.95,
                        "attributes": {
                            "color": "red",
                            "tracking_id": "vehicle_001",
                            "direction": "right"
                        }
                    },
                    {
                        "object": "truck",
                        "bbox": [400, 100, 600, 350],
                        "confidence": 0.92,
                        "attributes": {
                            "color": "white",
                            "tracking_id": "vehicle_002",
                            "direction": "left"
                        }
                    }
                ],
                "metadata": {
                    "frame_number": 1,
                    "is_keyframe": True
                }
            },
            keyframes_dir / "keyframe_0050.jpg": {
                "annotations": [
                    {
                        "object": "car",
                        "bbox": [450, 150, 600, 300],
                        "confidence": 0.93,
                        "attributes": {
                            "color": "red",
                            "tracking_id": "vehicle_001",
                            "direction": "right"
                        }
                    }
                ],
                "metadata": {
                    "frame_number": 50,
                    "is_keyframe": True
                }
            }
        }
        
        # Или загружаем из файлов
        # keyframe_annotations = await prepare_keyframe_annotations(keyframes_dir, agent)
        
        logger.info(f"Loaded {len(keyframe_annotations)} keyframe annotations")
        logger.info("Starting video annotation with keyframe guidance...")
        
        # Запуск разметки с учетом опорных кадров
        result = await agent.annotate_video_with_keyframes(
            video_path=video_path,
            task_description=task_description,
            keyframe_annotations=keyframe_annotations,
            output_dir=output_dir,
            max_frames=max_frames
        )
        
        # Вывод результатов
        logger.info(f"Annotation completed successfully!")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Total frames: {result['metadata']['total_frames']}")
        logger.info(f"Annotated frames: {result['metadata']['annotated_frames']}")
        logger.info(f"Keyframes used: {result['metadata']['keyframes_used']}")
        logger.info(f"Output directory: {result['output_dir']}")
        
        # Проверка консистентности tracking ID
        tracking_ids = set()
        for annotation in result['annotations'].values():
            for obj in annotation.get('annotations', []):
                if 'tracking_id' in obj.get('attributes', {}):
                    tracking_ids.add(obj['attributes']['tracking_id'])
        
        logger.info(f"\nUnique tracking IDs found: {len(tracking_ids)}")
        logger.info(f"Tracking IDs: {sorted(tracking_ids)}")
        
    except Exception as e:
        logger.error(f"Error during annotation: {e}")
        raise
    
    finally:
        await agent.cleanup()
        logger.info("Agent cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())
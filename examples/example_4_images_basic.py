"""
Пример 4: Базовая разметка изображений без примеров

Использование:
python examples/example_4_images_basic.py
"""

import asyncio
from pathlib import Path
from src.agents.image_annotation_agent import ImageAnnotationAgent
from src.utils.logging_config import setup_logging
import json

# Настройка логирования
logger = setup_logging()

async def main():
    """
    Вариант 4: Базовая разметка изображений без примеров
    
    Входные данные:
    - Папка с неразмеченными изображениями
    - Описание задачи разметки
    
    Выходные данные:
    - JSON файлы с разметкой для каждого изображения
    """
    
    # Инициализация агента
    agent = ImageAnnotationAgent()
    
    # Параметры задачи
    images_dir = Path("data/images/raw")
    
    task_description = """
    Perform general scene understanding and object detection for street view images.
    
    Identify and annotate:
    
    1. Vehicles (cars, trucks, buses, motorcycles, bicycles)
       - Type
       - Bounding box
       - Approximate distance (near, medium, far)
       - Movement status (parked, moving)
    
    2. Pedestrians
       - Bounding box
       - Activity (walking, standing, crossing)
       - Group size if applicable
    
    3. Traffic infrastructure
       - Traffic lights (state: red/yellow/green)
       - Road signs (type if recognizable)
       - Crosswalks
       - Lane markings
    
    4. Buildings and structures
       - General type (residential, commercial, industrial)
       - Bounding box for main structures
    
    5. Environmental conditions
       - Weather (clear, cloudy, rainy, etc.)
       - Lighting (day, night, dusk, dawn)
       - Road conditions (dry, wet, snow)
    
    Provide confidence scores for all detections.
    Use JSON format with proper structure.
    """
    
    output_dir = Path("output/images_basic")
    
    try:
        logger.info("Starting basic image annotation...")
        
        # Запуск базовой разметки
        result = await agent.annotate_images_basic(
            images_dir=images_dir,
            task_description=task_description,
            output_dir=output_dir
        )
        
        # Вывод результатов
        logger.info(f"Annotation completed successfully!")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Total images: {result['metadata']['total_images']}")
        logger.info(f"Annotated images: {result['metadata']['annotated_images']}")
        logger.info(f"Output directory: {result['output_dir']}")
        
        # Подробный анализ результатов
        if result['annotations']:
            # Статистика по типам объектов
            object_types = {}
            confidence_scores = []
            weather_conditions = {}
            lighting_conditions = {}
            
            for img_name, annotation in result['annotations'].items():
                for obj in annotation.get('annotations', []):
                    obj_type = obj.get('object', 'unknown')
                    object_types[obj_type] = object_types.get(obj_type, 0) + 1
                    
                    if 'confidence' in obj:
                        confidence_scores.append(obj['confidence'])
                
                # Анализ метаданных
                metadata = annotation.get('metadata', {})
                if 'weather' in metadata:
                    weather = metadata['weather']
                    weather_conditions[weather] = weather_conditions.get(weather, 0) + 1
                if 'lighting' in metadata:
                    lighting = metadata['lighting']
                    lighting_conditions[lighting] = lighting_conditions.get(lighting, 0) + 1
            
            logger.info(f"\n=== Annotation Statistics ===")
            
            logger.info(f"\nObject Types Detected:")
            for obj_type, count in sorted(object_types.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {obj_type}: {count}")
            
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                logger.info(f"\nAverage Confidence Score: {avg_confidence:.3f}")
                logger.info(f"Min Confidence: {min(confidence_scores):.3f}")
                logger.info(f"Max Confidence: {max(confidence_scores):.3f}")
            
            if weather_conditions:
                logger.info(f"\nWeather Conditions:")
                for weather, count in weather_conditions.items():
                    logger.info(f"  - {weather}: {count} images")
            
            if lighting_conditions:
                logger.info(f"\nLighting Conditions:")
                for lighting, count in lighting_conditions.items():
                    logger.info(f"  - {lighting}: {count} images")
            
            # Показать полную аннотацию первого изображения
            first_img_name = list(result['annotations'].keys())[0]
            first_annotation = result['annotations'][first_img_name]
            
            logger.info(f"\n=== Example Full Annotation ===")
            logger.info(f"Image: {first_img_name}")
            logger.info(json.dumps(first_annotation, indent=2, ensure_ascii=False))
            
        else:
            logger.warning("No annotations were created!")
        
        # Сохранение сводной статистики
        summary_path = Path(result['output_dir']) / "summary_statistics.json"
        summary = {
            "total_images": result['metadata']['total_images'],
            "annotated_images": result['metadata']['annotated_images'],
            "object_types": object_types,
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "weather_distribution": weather_conditions,
            "lighting_distribution": lighting_conditions
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nSummary statistics saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"Error during annotation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    finally:
        await agent.cleanup()
        logger.info("Agent cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())
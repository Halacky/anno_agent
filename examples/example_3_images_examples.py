"""
Пример 3: Разметка изображений с использованием примеров

Использование:
python examples/example_3_images_examples.py
"""

import asyncio
from pathlib import Path
from src.agents.image_annotation_agent import ImageAnnotationAgent
from src.utils.logging_config import setup_logging
from src.utils.file_utils import save_annotation

# Настройка логирования
logger = setup_logging()

def create_example_annotations(examples_dir: Path):
    """
    Создание примеров аннотаций для обучения агента
    
    В реальном сценарии эти аннотации создаются вручную
    или импортируются из существующей размеченной базы
    """
    
    # Пример 1: Размеченное изображение продукта
    example1_annotation = {
        "annotations": [
            {
                "object": "product_box",
                "bbox": [50, 30, 450, 470],
                "confidence": 1.0,
                "attributes": {
                    "category": "electronics",
                    "brand_visible": True,
                    "condition": "new",
                    "packaging": "sealed"
                }
            },
            {
                "object": "brand_logo",
                "bbox": [120, 80, 280, 140],
                "confidence": 1.0,
                "attributes": {
                    "logo_name": "TechBrand",
                    "clarity": "high"
                }
            },
            {
                "object": "price_tag",
                "bbox": [380, 400, 440, 450],
                "confidence": 1.0,
                "attributes": {
                    "price": "$299",
                    "readable": True
                }
            }
        ],
        "metadata": {
            "annotation_type": "product_catalog",
            "annotator": "manual",
            "quality": "high"
        }
    }
    
    # Пример 2: Еще одно размеченное изображение
    example2_annotation = {
        "annotations": [
            {
                "object": "product_box",
                "bbox": [80, 50, 520, 480],
                "confidence": 1.0,
                "attributes": {
                    "category": "clothing",
                    "brand_visible": True,
                    "condition": "new",
                    "packaging": "box"
                }
            },
            {
                "object": "brand_logo",
                "bbox": [180, 100, 340, 160],
                "confidence": 1.0,
                "attributes": {
                    "logo_name": "FashionCo",
                    "clarity": "medium"
                }
            }
        ],
        "metadata": {
            "annotation_type": "product_catalog",
            "annotator": "manual",
            "quality": "high"
        }
    }
    
    # Сохранение примеров (если файлы существуют)
    example_files = list(examples_dir.glob("*.jpg")) + list(examples_dir.glob("*.png"))
    
    if len(example_files) >= 1:
        save_annotation(
            example1_annotation, 
            examples_dir / f"{example_files[0].stem}_annotation.json"
        )
        logger.info(f"Created annotation for {example_files[0].name}")
    
    if len(example_files) >= 2:
        save_annotation(
            example2_annotation, 
            examples_dir / f"{example_files[1].stem}_annotation.json"
        )
        logger.info(f"Created annotation for {example_files[1].name}")

async def main():
    """
    Вариант 3: Разметка изображений с примерами
    
    Входные данные:
    - Папка с неразмеченными изображениями
    - Папка с примерами разметки
    - Описание задачи разметки
    
    Выходные данные:
    - JSON файлы с разметкой для каждого изображения
    """
    
    # Инициализация агента
    agent = ImageAnnotationAgent()
    
    # Параметры задачи
    images_dir = Path("data/images/unlabeled")
    examples_dir = Path("data/images/examples")
    
    task_description = """
    Annotate product images for e-commerce catalog.
    For each product image, identify and annotate:
    
    1. Product box/item (main object)
       - Bounding box
       - Category (electronics, clothing, toys, etc.)
       - Condition (new, used, refurbished)
       - Packaging type (sealed, box, blister, etc.)
    
    2. Brand logo (if visible)
       - Bounding box
       - Logo name if recognizable
       - Clarity level (high, medium, low)
    
    3. Price tag (if visible)
       - Bounding box
       - Price value if readable
       - Readability flag
    
    4. Any other relevant product features
    
    Follow the annotation style and structure from the provided examples.
    Maintain consistency in attribute naming and categorization.
    """
    
    output_dir = Path("output/images_with_examples")
    
    try:
        logger.info("Setting up example annotations...")
        
        # Создание примеров аннотаций (в реальном случае они уже существуют)
        examples_dir.mkdir(parents=True, exist_ok=True)
        create_example_annotations(examples_dir)
        
        logger.info("Starting image annotation with examples...")
        
        # Запуск разметки с использованием примеров
        result = await agent.annotate_images_with_examples(
            images_dir=images_dir,
            examples_dir=examples_dir,
            task_description=task_description,
            output_dir=output_dir
        )
        
        # Вывод результатов
        logger.info(f"Annotation completed successfully!")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Total images: {result['metadata']['total_images']}")
        logger.info(f"Annotated images: {result['metadata']['annotated_images']}")
        logger.info(f"Examples used: {result['metadata']['examples_used']}")
        logger.info(f"Output directory: {result['output_dir']}")
        
        # Анализ результатов
        if result['annotations']:
            categories_found = {}
            total_objects = 0
            
            for img_name, annotation in result['annotations'].items():
                for obj in annotation.get('annotations', []):
                    total_objects += 1
                    category = obj.get('attributes', {}).get('category', 'unknown')
                    categories_found[category] = categories_found.get(category, 0) + 1
            
            logger.info(f"\nAnnotation statistics:")
            logger.info(f"Total objects detected: {total_objects}")
            logger.info(f"Categories distribution:")
            for category, count in sorted(categories_found.items()):
                logger.info(f"  - {category}: {count}")
            
            # Пример одной аннотации
            first_annotation = list(result['annotations'].values())[0]
            logger.info(f"\nExample annotation:")
            logger.info(f"Image: {first_annotation.get('image_path')}")
            logger.info(f"Objects found: {len(first_annotation.get('annotations', []))}")
            
    except Exception as e:
        logger.error(f"Error during annotation: {e}")
        raise
    
    finally:
        await agent.cleanup()
        logger.info("Agent cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())
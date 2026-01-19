"""
SAM3 + Qwen VLM Agent
Интегрированный агент для визуального анализа и сегментации объектов
"""

import requests
import base64
import json
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from io import BytesIO
from PIL import Image
import numpy as np


@dataclass
class BoundingBox:
    """Нормализованный bounding box [cx, cy, w, h]"""
    cx: float
    cy: float
    w: float
    h: float
    
    def to_list(self) -> List[float]:
        return [self.cx, self.cy, self.w, self.h]
    
    def to_xyxy(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Конвертация в абсолютные координаты [x1, y1, x2, y2]"""
        x1 = int((self.cx - self.w / 2) * width)
        y1 = int((self.cy - self.h / 2) * height)
        x2 = int((self.cx + self.w / 2) * width)
        y2 = int((self.cy + self.h / 2) * height)
        return (x1, y1, x2, y2)


@dataclass
class DetectedObject:
    """Обнаруженный объект с атрибутами"""
    category: str
    bbox: BoundingBox
    confidence: float
    attributes: Optional[Dict] = None


@dataclass
class SegmentationResult:
    """Результат сегментации"""
    mask: str  # RLE-encoded mask
    bbox: BoundingBox
    score: float
    object_info: Optional[DetectedObject] = None


class SAM3QwenAgent:
    """
    Агент для совместного использования SAM3 и Qwen VLM.
    
    Workflow:
    1. Qwen анализирует изображение и определяет объекты
    2. SAM3 выполняет точную сегментацию на основе результатов Qwen
    """
    
    def __init__(
        self,
        sam3_url: str = "http://localhost:8000",
        qwen_url: str = "http://localhost:8001",
        sam3_api_version: str = "v1",
        qwen_api_version: str = "v1"
    ):
        self.sam3_url = sam3_url.rstrip('/')
        self.qwen_url = qwen_url.rstrip('/')
        self.sam3_api_version = sam3_api_version
        self.qwen_api_version = qwen_api_version
        
    def _image_to_base64(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """Конвертация изображения в base64"""
        if isinstance(image, str):
            # Если это путь к файлу
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image, Image.Image):
            # PIL Image
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(image, np.ndarray):
            # NumPy array
            img = Image.fromarray(image)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            raise ValueError("Unsupported image type")
    
    def detect_objects_with_qwen(
        self,
        image: Union[str, Image.Image, np.ndarray],
        categories: Optional[List[str]] = None,
        custom_prompt: Optional[str] = None,
        include_attributes: bool = True
    ) -> List[DetectedObject]:
        """
        Обнаружение объектов с помощью Qwen VLM
        
        Args:
            image: Путь к файлу, PIL Image или NumPy array
            categories: Список категорий для поиска (если None, то все объекты)
            custom_prompt: Кастомный промпт для Qwen
            include_attributes: Включать ли атрибуты объектов
            
        Returns:
            Список обнаруженных объектов с bounding boxes
        """
        image_b64 = self._image_to_base64(image)
        
        # Формируем запрос к Qwen для 2D grounding
        payload = {
            "image_base64": image_b64,
            "include_attributes": include_attributes
        }
        
        if categories:
            payload["categories"] = categories
        
        if custom_prompt:
            payload["prompt"] = custom_prompt
        
        url = f"{self.qwen_url}/api/{self.qwen_api_version}/grounding/2d"
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Парсим результаты
        detected_objects = []
        for detection in result.get('detections', []):
            bbox_data = detection.get('bbox', {})
            bbox = BoundingBox(
                cx=bbox_data.get('cx', 0),
                cy=bbox_data.get('cy', 0),
                w=bbox_data.get('w', 0),
                h=bbox_data.get('h', 0)
            )
            
            obj = DetectedObject(
                category=detection.get('category', 'unknown'),
                bbox=bbox,
                confidence=detection.get('confidence', 0.0),
                attributes=detection.get('attributes')
            )
            detected_objects.append(obj)
        
        return detected_objects
    
    def segment_with_sam3(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompts: List[Dict],
        confidence_threshold: float = 0.5
    ) -> List[SegmentationResult]:
        """
        Сегментация с помощью SAM3
        
        Args:
            image: Изображение
            prompts: Список промптов для SAM3
            confidence_threshold: Порог уверенности
            
        Returns:
            Список результатов сегментации
        """
        image_b64 = self._image_to_base64(image)
        
        payload = {
            "image": image_b64,
            "prompts": prompts,
            "confidence_threshold": confidence_threshold
        }
        
        url = f"{self.sam3_url}/api/{self.sam3_api_version}/image/segment"
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Парсим результаты
        segmentations = []
        for i in range(result['num_masks']):
            bbox_list = result['boxes'][i]
            bbox = BoundingBox(
                cx=bbox_list[0],
                cy=bbox_list[1],
                w=bbox_list[2],
                h=bbox_list[3]
            )
            
            seg = SegmentationResult(
                mask=result['masks'][i],
                bbox=bbox,
                score=result['scores'][i]
            )
            segmentations.append(seg)
        
        return segmentations
    
    def analyze_and_segment(
        self,
        image: Union[str, Image.Image, np.ndarray],
        query: str,
        categories: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        use_text_prompts: bool = True
    ) -> List[SegmentationResult]:
        """
        Полный пайплайн: анализ с Qwen + сегментация с SAM3
        
        Args:
            image: Изображение для анализа
            query: Запрос (например, "найди все машины")
            categories: Список категорий для поиска
            confidence_threshold: Порог уверенности для SAM3
            use_text_prompts: Использовать текстовые промпты для SAM3
            
        Returns:
            Список результатов сегментации с информацией об объектах
        """
        print(f"🔍 Анализ изображения с Qwen VLM...")
        print(f"   Запрос: {query}")
        
        # Шаг 1: Обнаружение объектов с Qwen
        detected_objects = self.detect_objects_with_qwen(
            image=image,
            categories=categories,
            custom_prompt=query,
            include_attributes=True
        )
        
        print(f"✅ Обнаружено объектов: {len(detected_objects)}")
        for obj in detected_objects:
            print(f"   - {obj.category} (confidence: {obj.confidence:.2f})")
        
        if not detected_objects:
            print("⚠️  Объекты не найдены")
            return []
        
        # Шаг 2: Подготовка промптов для SAM3
        sam3_prompts = []
        
        for obj in detected_objects:
            if use_text_prompts:
                # Используем текстовый промпт с атрибутами
                text_description = obj.category
                if obj.attributes:
                    attrs = ", ".join([f"{k}: {v}" for k, v in obj.attributes.items()])
                    text_description = f"{obj.category} ({attrs})"
                
                sam3_prompts.append({
                    "type": "text",
                    "text": text_description
                })
            else:
                # Используем bounding box промпт
                sam3_prompts.append({
                    "type": "box",
                    "box": obj.bbox.to_list(),
                    "label": True
                })
        
        print(f"\n🎯 Сегментация с SAM3...")
        print(f"   Промптов: {len(sam3_prompts)}")
        
        # Шаг 3: Сегментация с SAM3
        segmentations = self.segment_with_sam3(
            image=image,
            prompts=sam3_prompts,
            confidence_threshold=confidence_threshold
        )
        
        # Связываем сегментации с обнаруженными объектами
        for i, seg in enumerate(segmentations):
            if i < len(detected_objects):
                seg.object_info = detected_objects[i]
        
        print(f"✅ Сегментировано объектов: {len(segmentations)}")
        for seg in segmentations:
            if seg.object_info:
                print(f"   - {seg.object_info.category} (score: {seg.score:.2f})")
        
        return segmentations
    
    def interactive_segment(
        self,
        image: Union[str, Image.Image, np.ndarray],
        description: str,
        detail_level: str = "detailed"
    ) -> List[SegmentationResult]:
        """
        Интерактивная сегментация: сначала получаем детальное описание,
        затем сегментируем найденные объекты
        
        Args:
            image: Изображение
            description: Описание того, что нужно найти
            detail_level: Уровень детализации анализа
            
        Returns:
            Результаты сегментации
        """
        image_b64 = self._image_to_base64(image)
        
        # Шаг 1: Получаем детальное описание сцены
        print(f"🔍 Анализ сцены с Qwen VLM...")
        
        desc_url = f"{self.qwen_url}/api/{self.qwen_api_version}/image/description"
        desc_response = requests.post(desc_url, json={
            "image_base64": image_b64,
            "detail_level": detail_level,
            "prompt": f"Describe the image and identify: {description}"
        })
        desc_response.raise_for_status()
        
        scene_description = desc_response.json().get('description', '')
        print(f"📝 Описание сцены:\n{scene_description}\n")
        
        # Шаг 2: Выполняем сегментацию
        return self.analyze_and_segment(
            image=image,
            query=description,
            use_text_prompts=True
        )
    
    def spatial_segment(
        self,
        image: Union[str, Image.Image, np.ndarray],
        spatial_query: str
    ) -> List[SegmentationResult]:
        """
        Сегментация с учетом пространственных отношений
        
        Args:
            image: Изображение
            spatial_query: Пространственный запрос (например, "объекты на столе")
            
        Returns:
            Результаты сегментации
        """
        image_b64 = self._image_to_base64(image)
        
        print(f"🔍 Пространственный анализ с Qwen VLM...")
        print(f"   Запрос: {spatial_query}")
        
        # Получаем пространственное понимание
        spatial_url = f"{self.qwen_url}/api/{self.qwen_api_version}/spatial/understanding"
        spatial_response = requests.post(spatial_url, json={
            "image_base64": image_b64,
            "query": spatial_query,
            "prompt": spatial_query
        })
        spatial_response.raise_for_status()
        
        spatial_result = spatial_response.json()
        print(f"📝 Результат анализа:\n{spatial_result.get('answer', '')}\n")
        
        # Выполняем сегментацию найденных объектов
        return self.analyze_and_segment(
            image=image,
            query=spatial_query,
            use_text_prompts=True
        )


# ==================== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ====================

def example_basic_usage():
    """Базовый пример использования"""
    print("=" * 60)
    print("ПРИМЕР 1: Базовое использование")
    print("=" * 60)
    
    agent = SAM3QwenAgent(
        sam3_url="http://localhost:8000",
        qwen_url="http://localhost:8001"
    )
    
    # Анализ и сегментация
    results = agent.analyze_and_segment(
        image="path/to/image.jpg",
        query="найди всех людей на изображении",
        categories=["person"],
        confidence_threshold=0.6
    )
    
    print(f"\n📊 Результаты:")
    for i, result in enumerate(results):
        print(f"\nОбъект {i+1}:")
        print(f"  Категория: {result.object_info.category}")
        print(f"  Уверенность: {result.score:.2f}")
        print(f"  BBox: {result.bbox.to_list()}")
        if result.object_info.attributes:
            print(f"  Атрибуты: {result.object_info.attributes}")


def example_multi_category():
    """Пример с несколькими категориями"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 2: Поиск нескольких категорий объектов")
    print("=" * 60)
    
    agent = SAM3QwenAgent()
    
    results = agent.analyze_and_segment(
        image="street_scene.jpg",
        query="найди все транспортные средства и пешеходов",
        categories=["person", "car", "bicycle", "motorcycle"],
        use_text_prompts=True
    )
    
    # Группируем по категориям
    by_category = {}
    for result in results:
        cat = result.object_info.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(result)
    
    print(f"\n📊 Статистика по категориям:")
    for category, items in by_category.items():
        print(f"  {category}: {len(items)} объектов")


def example_interactive():
    """Интерактивная сегментация с детальным описанием"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 3: Интерактивная сегментация")
    print("=" * 60)
    
    agent = SAM3QwenAgent()
    
    results = agent.interactive_segment(
        image="room.jpg",
        description="мебель и электроника",
        detail_level="comprehensive"
    )
    
    print(f"\n📊 Найдено объектов: {len(results)}")


def example_spatial():
    """Пространственная сегментация"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 4: Пространственная сегментация")
    print("=" * 60)
    
    agent = SAM3QwenAgent()
    
    results = agent.spatial_segment(
        image="office.jpg",
        spatial_query="какие предметы находятся на рабочем столе?"
    )
    
    print(f"\n📊 Найдено объектов на столе: {len(results)}")


def example_advanced_workflow():
    """Продвинутый workflow с обработкой результатов"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 5: Продвинутый workflow")
    print("=" * 60)
    
    agent = SAM3QwenAgent()
    
    # Шаг 1: Детальный анализ
    image = "complex_scene.jpg"
    
    # Получаем общее описание
    image_b64 = agent._image_to_base64(image)
    desc_response = requests.post(
        f"{agent.qwen_url}/api/v1/image/description",
        json={
            "image_base64": image_b64,
            "detail_level": "comprehensive"
        }
    )
    description = desc_response.json().get('description', '')
    print(f"📝 Описание сцены:\n{description}\n")
    
    # Шаг 2: Сегментация специфических объектов
    results = agent.analyze_and_segment(
        image=image,
        query="найди все объекты красного цвета",
        use_text_prompts=True,
        confidence_threshold=0.7
    )
    
    # Шаг 3: Фильтрация по score
    high_confidence_results = [r for r in results if r.score > 0.8]
    
    print(f"\n📊 Результаты:")
    print(f"  Всего найдено: {len(results)}")
    print(f"  Высокая уверенность (>0.8): {len(high_confidence_results)}")
    
    # Шаг 4: Сохранение результатов
    output = {
        "description": description,
        "total_objects": len(results),
        "objects": [
            {
                "category": r.object_info.category,
                "score": r.score,
                "bbox": r.bbox.to_list(),
                "attributes": r.object_info.attributes
            }
            for r in results
        ]
    }
    
    print(f"\n💾 Результаты сохранены в JSON")
    print(json.dumps(output, indent=2, ensure_ascii=False))


def example_bbox_vs_text():
    """Сравнение bounding box и текстовых промптов"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 6: BBox промпты vs Текстовые промпты")
    print("=" * 60)
    
    agent = SAM3QwenAgent()
    
    image = "test_image.jpg"
    
    # С bounding box промптами
    print("\n🔹 Использование BBox промптов:")
    results_bbox = agent.analyze_and_segment(
        image=image,
        query="найди машины",
        categories=["car"],
        use_text_prompts=False  # Используем bbox
    )
    
    # С текстовыми промптами
    print("\n🔹 Использование текстовых промптов:")
    results_text = agent.analyze_and_segment(
        image=image,
        query="найди машины",
        categories=["car"],
        use_text_prompts=True  # Используем text
    )
    
    print(f"\n📊 Сравнение результатов:")
    print(f"  BBox промпты: {len(results_bbox)} объектов")
    print(f"  Текстовые промпты: {len(results_text)} объектов")


if __name__ == "__main__":
    """
    Запуск примеров использования
    
    Перед запуском убедитесь, что:
    1. SAM3 API запущен на http://localhost:8000
    2. Qwen VLM API запущен на http://localhost:8001
    """
    
    # Раскомментируйте нужный пример:
    
    # example_basic_usage()
    # example_multi_category()
    # example_interactive()
    # example_spatial()
    # example_advanced_workflow()
    # example_bbox_vs_text()
    
    print("\n✅ Все примеры готовы к использованию!")
    print("Раскомментируйте нужный пример в блоке if __name__ == '__main__'")

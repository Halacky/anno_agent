"""
SAM3 + Qwen VLM Agent
Интегрированный агент для визуального анализа и сегментации объектов
"""

import requests
import base64
import json
import re
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
    
    @classmethod
    def from_xyxy_normalized(cls, x1: float, y1: float, x2: float, y2: float):
        """Создать из нормализованных координат [x1, y1, x2, y2]"""
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return cls(cx, cy, w, h)
    
    @classmethod
    def from_xyxy_absolute(cls, x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int):
        """Создать из абсолютных координат [x1, y1, x2, y2] в пикселях"""
        x1_norm = x1 / img_width
        y1_norm = y1 / img_height
        x2_norm = x2 / img_width
        y2_norm = y2 / img_height
        return cls.from_xyxy_normalized(x1_norm, y1_norm, x2_norm, y2_norm)


@dataclass
class DetectedObject:
    """Обнаруженный объект с атрибутами"""
    category: str
    bbox: BoundingBox
    confidence: float
    attributes: Optional[Dict] = None
    text_description: Optional[str] = None


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
    1. Qwen анализирует изображение и определяет объекты через /v1/grounding/2d
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
    
    def _get_image_size(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[int, int]:
        """Получить размеры изображения (width, height)"""
        if isinstance(image, str):
            with Image.open(image) as img:
                return img.size
        elif isinstance(image, Image.Image):
            return image.size
        elif isinstance(image, np.ndarray):
            # NumPy array в формате (height, width, channels)
            return (image.shape[1], image.shape[0])
        else:
            raise ValueError("Unsupported image type")
    
    def _parse_grounding_response(self, response_data: Dict, img_width: int, img_height: int) -> List[DetectedObject]:
        """
        Парсинг ответа Qwen grounding/2d
        
        Ожидаемый формат в response_data["result"]:
        {
            "detections": [
                {
                    "label": "container",
                    "bbox": [x1, y1, x2, y2],  # абсолютные координаты в пикселях
                    "confidence": 0.95
                }
            ]
        }
        """
        detected_objects = []
        
        # Извлекаем результат
        result = response_data.get('result', response_data.get('data', {}))
        
        # Проверяем разные возможные структуры ответа
        detections = result.get('detections', [])
        
        # Если detections пустой, попробуем другие поля
        if not detections:
            # Возможно результат в других полях
            if isinstance(result, list):
                detections = result
            elif 'objects' in result:
                detections = result['objects']
        
        print(f"🔍 Найдено детекций в ответе: {len(detections)}")
        
        for detection in detections:
            # Извлекаем метку/категорию
            label = detection.get('label') or detection.get('category') or detection.get('class', 'unknown')
            
            # Извлекаем bbox - может быть в разных форматах
            bbox_raw = detection.get('bbox') or detection.get('box') or detection.get('bounding_box')
            
            if not bbox_raw:
                print(f"⚠️  Пропускаем детекцию без bbox: {detection}")
                continue
            
            # bbox может быть списком [x1, y1, x2, y2] или словарём
            if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                x1, y1, x2, y2 = bbox_raw
            elif isinstance(bbox_raw, dict):
                x1 = bbox_raw.get('x1', bbox_raw.get('xmin', 0))
                y1 = bbox_raw.get('y1', bbox_raw.get('ymin', 0))
                x2 = bbox_raw.get('x2', bbox_raw.get('xmax', 0))
                y2 = bbox_raw.get('y2', bbox_raw.get('ymax', 0))
            else:
                print(f"⚠️  Неизвестный формат bbox: {bbox_raw}")
                continue
            
            # Конвертируем в нормализованный формат
            bbox = BoundingBox.from_xyxy_absolute(
                int(x1), int(y1), int(x2), int(y2),
                img_width, img_height
            )
            
            # Извлекаем уверенность
            confidence = detection.get('confidence', detection.get('score', 1.0))
            
            # Извлекаем атрибуты если есть
            attributes = detection.get('attributes')
            
            obj = DetectedObject(
                category=label,
                bbox=bbox,
                confidence=float(confidence),
                attributes=attributes,
                text_description=f"{label}"
            )
            
            detected_objects.append(obj)
            print(f"   ✓ {label}: bbox={bbox.to_list()}, conf={confidence:.2f}")
        
        return detected_objects
    
    def detect_objects_with_qwen(
        self,
        image: Union[str, Image.Image, np.ndarray],
        categories: Optional[List[str]] = None,
        custom_prompt: Optional[str] = None
    ) -> List[DetectedObject]:
        """
        Обнаружение объектов с помощью Qwen VLM используя /v1/grounding/2d endpoint
        
        Args:
            image: Путь к файлу, PIL Image или NumPy array
            categories: Список категорий для поиска
            custom_prompt: Кастомный промпт для Qwen
            
        Returns:
            Список обнаруженных объектов с bounding boxes
        """
        image_b64 = self._image_to_base64(image)
        img_width, img_height = self._get_image_size(image)
        
        # Формируем промпт для детекции
        if custom_prompt:
            prompt = custom_prompt
        elif categories:
            # Формат: "Detect all objects: cat1, cat2, cat3"
            cats_str = ", ".join(categories)
            prompt = f"Detect all objects: {cats_str}"
        else:
            prompt = "Detect all objects in the image"
        
        print(f"🔍 Qwen grounding/2d запрос: {prompt}")
        
        # Формируем payload согласно API
        payload = {
            "image_base64": image_b64,
            "prompt": prompt
        }
        
        url = f"{self.qwen_url}/{self.qwen_api_version}/grounding/2d"
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Не удалось подключиться к Qwen API по адресу {url}. "
                f"Убедитесь, что сервер запущен на {self.qwen_url}"
            ) from e
        
        result = response.json()
        
        # Выводим сырой ответ для отладки
        print(f"📝 Сырой ответ от Qwen:")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:1000])
        
        # Парсим результат
        detected_objects = self._parse_grounding_response(result, img_width, img_height)
        
        if not detected_objects:
            print("⚠️  Не удалось извлечь объекты из ответа Qwen")
            print(f"   Структура ответа: {list(result.keys())}")
        
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
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Не удалось подключиться к SAM3 API по адресу {url}. "
                f"Убедитесь, что сервер запущен на {self.sam3_url}"
            ) from e
        
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
        use_text_prompts: bool = True,
        qwen_confidence_threshold: float = 0.3
    ) -> List[SegmentationResult]:
        """
        Полный пайплайн: анализ с Qwen + сегментация с SAM3
        
        Args:
            image: Изображение для анализа
            query: Запрос (например, "найди все машины")
            categories: Список категорий для поиска
            confidence_threshold: Порог уверенности для SAM3
            use_text_prompts: Использовать текстовые промпты для SAM3
            qwen_confidence_threshold: Порог уверенности для фильтрации детекций Qwen
            
        Returns:
            Список результатов сегментации с информацией об объектах
        """
        print("=" * 70)
        print(f"🔍 Анализ изображения с Qwen VLM...")
        print(f"   Запрос: {query}")
        if categories:
            print(f"   Категории: {', '.join(categories)}")
        print("=" * 70)
        
        # Шаг 1: Обнаружение объектов с Qwen
        detected_objects = self.detect_objects_with_qwen(
            image=image,
            categories=categories,
            custom_prompt=query
        )
        
        # Фильтруем по уверенности
        detected_objects = [
            obj for obj in detected_objects 
            if obj.confidence >= qwen_confidence_threshold
        ]
        
        print(f"\n✅ Обнаружено объектов после фильтрации (conf >= {qwen_confidence_threshold}): {len(detected_objects)}")
        for obj in detected_objects:
            print(f"   - {obj.category} (confidence: {obj.confidence:.2f})")
        
        if not detected_objects:
            print("⚠️  Объекты не найдены или не прошли порог уверенности")
            return []
        
        # Шаг 2: Подготовка промптов для SAM3
        sam3_prompts = []
        
        for obj in detected_objects:
            if use_text_prompts:
                # Используем текстовый промпт
                text_description = obj.text_description or obj.category
                
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
        print(f"   Тип промптов: {'text' if use_text_prompts else 'box'}")
        
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
        
        print(f"\n✅ Сегментировано объектов: {len(segmentations)}")
        for seg in segmentations:
            if seg.object_info:
                print(f"   - {seg.object_info.category} (SAM3 score: {seg.score:.2f})")
        
        print("=" * 70)
        
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
        
        desc_url = f"{self.qwen_url}/{self.qwen_api_version}/image/description"
        desc_response = requests.post(desc_url, json={
            "image_base64": image_b64,
            "detail_level": detail_level,
            "prompt": f"Describe the image and identify: {description}"
        }, timeout=60)
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
        spatial_url = f"{self.qwen_url}/{self.qwen_api_version}/spatial/understanding"
        
        try:
            spatial_response = requests.post(spatial_url, json={
                "image_base64": image_b64,
                "query": spatial_query,
                "prompt": spatial_query
            }, timeout=60)
            spatial_response.raise_for_status()
            
            spatial_result = spatial_response.json()
            print(f"📝 Результат анализа:\n{spatial_result.get('answer', '')}\n")
        except requests.exceptions.HTTPError:
            print("⚠️  Endpoint spatial/understanding недоступен, используем grounding/2d")
        
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
    
    # ВАЖНО: Укажите реальный путь к изображению
    image_path = "/home/golovanks/projects/sgp_kras/MainHanlder/CT/anno_agent-main/tmp_cvat_download/images/00001.jpg"
    
    try:
        # Анализ и сегментация контейнеров
        results = agent.analyze_and_segment(
            image=image_path,
            query="Detect all objects: containers",
            categories=["container"],
            confidence_threshold=0.5,
            qwen_confidence_threshold=0.3
        )
        
        print(f"\n📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"Всего сегментировано: {len(results)} объектов\n")
        
        for i, result in enumerate(results):
            print(f"Объект {i+1}:")
            if result.object_info:
                print(f"  Категория: {result.object_info.category}")
                print(f"  Qwen confidence: {result.object_info.confidence:.2f}")
                if result.object_info.text_description:
                    print(f"  Описание: {result.object_info.text_description}")
            print(f"  SAM3 score: {result.score:.2f}")
            print(f"  BBox (normalized): {result.bbox.to_list()}")
            print()
            
    except ConnectionError as e:
        print(f"\n❌ Ошибка подключения: {e}")
        print("\nПроверьте, что серверы запущены:")
        print("  - SAM3 API: http://localhost:8000")
        print("  - Qwen API: http://localhost:8001")
    except FileNotFoundError:
        print(f"\n❌ Файл не найден: {image_path}")
        print("Укажите правильный путь к изображению")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


def example_bbox_vs_text():
    """Сравнение bbox и text промптов"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 2: Сравнение BBox vs Text промптов")
    print("=" * 60)
    
    agent = SAM3QwenAgent()
    image_path = "/home/golovanks/projects/sgp_kras/MainHanlder/CT/anno_agent-main/tmp_cvat_download/images/00001.jpg"
    
    try:
        # Тест с bbox промптами
        print("\n🔹 ТЕСТ 1: BBox промпты")
        results_bbox = agent.analyze_and_segment(
            image=image_path,
            query="Detect all objects: containers",
            categories=["container"],
            use_text_prompts=False  # BBox
        )
        
        # Тест с text промптами
        print("\n🔹 ТЕСТ 2: Text промпты")
        results_text = agent.analyze_and_segment(
            image=image_path,
            query="Detect all objects: containers",
            categories=["container"],
            use_text_prompts=True  # Text
        )
        
        print(f"\n📊 СРАВНЕНИЕ:")
        print(f"  BBox промпты: {len(results_bbox)} объектов")
        print(f"  Text промпты: {len(results_text)} объектов")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


def example_simple_test():
    """Простой тест подключения"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Проверка подключения")
    print("=" * 60)
    
    agent = SAM3QwenAgent()
    
    # Проверка Qwen API
    print("\n1️⃣ Проверка Qwen API...")
    try:
        response = requests.get(f"{agent.qwen_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Qwen API доступен")
        else:
            print(f"   ⚠️  Qwen API вернул код: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Qwen API недоступен: {e}")
    
    # Проверка SAM3 API
    print("\n2️⃣ Проверка SAM3 API...")
    try:
        response = requests.get(f"{agent.sam3_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ SAM3 API доступен")
        else:
            print(f"   ⚠️  SAM3 API вернул код: {response.status_code}")
    except Exception as e:
        print(f"   ❌ SAM3 API недоступен: {e}")


if __name__ == "__main__":
    """
    Запуск примеров использования
    
    Перед запуском убедитесь, что:
    1. SAM3 API запущен на http://localhost:8000
    2. Qwen VLM API запущен на http://localhost:8001
    """
    
    # Проверка подключения
    example_simple_test()
    
    # Основной пример
    example_basic_usage()
    
    # Сравнение промптов
    # example_bbox_vs_text()

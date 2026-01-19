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
    
    def _parse_grounding_response(self, text: str) -> List[DetectedObject]:
        """
        Парсинг ответа Qwen с координатами объектов
        Формат: <|object_ref_start|>объект<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
        """
        detected_objects = []
        
        # Паттерн для извлечения объектов и их координат
        pattern = r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
        matches = re.findall(pattern, text)
        
        for match in matches:
            object_name = match[0].strip()
            x1, y1, x2, y2 = map(int, match[1:5])
            
            # Предполагаем, что координаты в пикселях и нужно нормализовать
            # Для этого нужны размеры изображения, но пока используем как есть
            # В реальном использовании нужно передавать размеры изображения
            
            bbox = BoundingBox.from_xyxy_normalized(
                x1 / 1000.0,  # Примерная нормализация
                y1 / 1000.0,
                x2 / 1000.0,
                y2 / 1000.0
            )
            
            obj = DetectedObject(
                category=object_name,
                bbox=bbox,
                confidence=1.0,  # Qwen не возвращает confidence
                text_description=object_name
            )
            detected_objects.append(obj)
        
        return detected_objects
    
    def detect_objects_with_qwen(
        self,
        image: Union[str, Image.Image, np.ndarray],
        categories: Optional[List[str]] = None,
        custom_prompt: Optional[str] = None
    ) -> List[DetectedObject]:
        """
        Обнаружение объектов с помощью Qwen VLM используя image/description endpoint
        
        Args:
            image: Путь к файлу, PIL Image или NumPy array
            categories: Список категорий для поиска (если None, то все объекты)
            custom_prompt: Кастомный промпт для Qwen
            
        Returns:
            Список обнаруженных объектов с описаниями
        """
        image_b64 = self._image_to_base64(image)
        
        # Формируем промпт для детекции объектов
        if custom_prompt:
            prompt = custom_prompt
        elif categories:
            cats_str = ", ".join(categories)
            prompt = f"Identify and locate all instances of the following objects in the image: {cats_str}. For each object, provide its location and description."
        else:
            prompt = "Identify and describe all significant objects in the image with their locations."
        
        # Используем image/description endpoint с детальным анализом
        payload = {
            "image_base64": image_b64,
            "detail_level": "comprehensive",
            "prompt": prompt
        }
        
        url = f"{self.qwen_url}/api/{self.qwen_api_version}/image/description"
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Не удалось подключиться к Qwen API по адресу {url}. "
                f"Убедитесь, что сервер запущен на {self.qwen_url}"
            ) from e
        
        result = response.json()
        description = result.get('description', '')
        
        print(f"📝 Ответ Qwen:\n{description}\n")
        
        # Парсим описание для извлечения объектов
        # Создаем объекты на основе текстового описания
        detected_objects = []
        
        # Попытка извлечь координаты если есть
        grounding_objects = self._parse_grounding_response(description)
        if grounding_objects:
            return grounding_objects
        
        # Если координат нет, создаем объекты из текста
        # Разбиваем описание на предложения
        sentences = [s.strip() for s in description.split('.') if s.strip()]
        
        for i, sentence in enumerate(sentences):
            # Ищем упоминания категорий
            if categories:
                for category in categories:
                    if category.lower() in sentence.lower():
                        # Создаем объект без точных координат (будет использован text prompt)
                        obj = DetectedObject(
                            category=category,
                            bbox=BoundingBox(0.5, 0.5, 0.8, 0.8),  # Примерная область
                            confidence=0.8,
                            text_description=sentence
                        )
                        detected_objects.append(obj)
                        break
            else:
                # Создаем общий объект из описания
                obj = DetectedObject(
                    category=f"object_{i+1}",
                    bbox=BoundingBox(0.5, 0.5, 0.8, 0.8),
                    confidence=0.8,
                    text_description=sentence
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
            custom_prompt=query
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
                # Используем текстовый промпт - либо описание, либо категорию
                text_description = obj.text_description or obj.category
                
                sam3_prompts.append({
                    "type": "text",
                    "text": text_description
                })
            else:
                # Используем bounding box промпт (если есть точные координаты)
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
        spatial_url = f"{self.qwen_url}/api/{self.qwen_api_version}/spatial/understanding"
        
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
            print("⚠️  Endpoint spatial/understanding недоступен, используем image/description")
            spatial_response = requests.post(
                f"{self.qwen_url}/api/{self.qwen_api_version}/image/description",
                json={
                    "image_base64": image_b64,
                    "prompt": spatial_query,
                    "detail_level": "detailed"
                },
                timeout=60
            )
            spatial_result = spatial_response.json()
            print(f"📝 Результат анализа:\n{spatial_result.get('description', '')}\n")
        
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
    image_path = "path/to/image.jpg"
    
    try:
        # Анализ и сегментация
        results = agent.analyze_and_segment(
            image=image_path,
            query="найди всех людей на изображении",
            categories=["person"],
            confidence_threshold=0.6
        )
        
        print(f"\n📊 Результаты:")
        for i, result in enumerate(results):
            print(f"\nОбъект {i+1}:")
            if result.object_info:
                print(f"  Категория: {result.object_info.category}")
                if result.object_info.text_description:
                    print(f"  Описание: {result.object_info.text_description}")
            print(f"  Уверенность SAM3: {result.score:.2f}")
            print(f"  BBox: {result.bbox.to_list()}")
            
    except ConnectionError as e:
        print(f"\n❌ Ошибка подключения: {e}")
        print("\nПроверьте, что серверы запущены:")
        print("  - SAM3 API: http://localhost:8000")
        print("  - Qwen API: http://localhost:8001")
    except FileNotFoundError:
        print(f"\n❌ Файл не найден: {image_path}")
        print("Укажите правильный путь к изображению")


def example_multi_category():
    """Пример с несколькими категориями"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 2: Поиск нескольких категорий объектов")
    print("=" * 60)
    
    agent = SAM3QwenAgent()
    
    try:
        results = agent.analyze_and_segment(
            image="street_scene.jpg",
            query="найди все транспортные средства и пешеходов",
            categories=["person", "car", "bicycle", "motorcycle"],
            use_text_prompts=True
        )
        
        # Группируем по категориям
        by_category = {}
        for result in results:
            if result.object_info:
                cat = result.object_info.category
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(result)
        
        print(f"\n📊 Статистика по категориям:")
        for category, items in by_category.items():
            print(f"  {category}: {len(items)} объектов")
            
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


def example_interactive():
    """Интерактивная сегментация с детальным описанием"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 3: Интерактивная сегментация")
    print("=" * 60)
    
    agent = SAM3QwenAgent()
    
    try:
        results = agent.interactive_segment(
            image="room.jpg",
            description="мебель и электроника",
            detail_level="comprehensive"
        )
        
        print(f"\n📊 Найдено объектов: {len(results)}")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


def example_simple_test():
    """Простой тест подключения и базовой функциональности"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Проверка подключения и базовой работы")
    print("=" * 60)
    
    agent = SAM3QwenAgent()
    
    # Проверка Qwen API
    print("\n1️⃣ Проверка Qwen API...")
    try:
        response = requests.get(f"{agent.qwen_url}/api/health", timeout=5)
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
            print(f"   Ответ: {response.json()}")
        else:
            print(f"   ⚠️  SAM3 API вернул код: {response.status_code}")
    except Exception as e:
        print(f"   ❌ SAM3 API недоступен: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    """
    Запуск примеров использования
    
    Перед запуском убедитесь, что:
    1. SAM3 API запущен на http://localhost:8000
    2. Qwen VLM API запущен на http://localhost:8001
    """
    
    # Сначала проверим подключение
    example_simple_test()
    
    # Затем запустите нужный пример:
    # example_basic_usage()
    # example_multi_category()
    # example_interactive()
    
    print("\n✅ Для запуска примеров:")
    print("1. Раскомментируйте нужный пример выше")
    print("2. Укажите правильный путь к изображению")
    print("3. Убедитесь, что оба API сервера запущены")

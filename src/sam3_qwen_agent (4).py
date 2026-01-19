"""
SAM3 + Qwen VLM Agent
Агент, где Qwen VLM анализирует изображение и генерирует промпты для SAM3
Основано на примере sam3_agent.ipynb из репозитория facebookresearch/sam3
"""

import requests
import base64
import json
import re
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
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
class SAM3Prompt:
    """Промпт для SAM3, сгенерированный MLLM"""
    type: str  # "text", "box", or "point"
    text: Optional[str] = None
    box: Optional[List[float]] = None
    points: Optional[List[List[float]]] = None
    point_labels: Optional[List[int]] = None
    reasoning: Optional[str] = None  # Почему MLLM выбрал этот промпт


@dataclass
class SegmentationResult:
    """Результат сегментации с контекстом от MLLM"""
    mask: str  # RLE-encoded mask
    bbox: BoundingBox
    score: float
    query: str  # Исходный запрос
    llm_reasoning: Optional[str] = None  # Объяснение от MLLM
    prompt_used: Optional[SAM3Prompt] = None


class SAM3QwenAgent:
    """
    SAM3 Agent - MLLM (Qwen) использует SAM3 как инструмент для сегментации
    
    Workflow:
    1. Пользователь: "segment the leftmost child wearing blue vest"
    2. Qwen анализирует изображение и определяет как лучше сегментировать
    3. Qwen генерирует промпт(ы) для SAM3 (text или bbox)
    4. SAM3 выполняет сегментацию на основе промптов от Qwen
    """
    
    # Системный промпт для MLLM, чтобы он работал как агент SAM3
    AGENT_SYSTEM_PROMPT = """You are a vision AI assistant that uses SAM3 (Segment Anything Model 3) as a tool.

Your task is to analyze images and generate appropriate prompts for SAM3 to segment objects based on user queries.

SAM3 supports three types of prompts:
1. TEXT prompts: Simple text descriptions (e.g., "red car", "person wearing hat")
2. BOX prompts: Bounding boxes [cx, cy, w, h] in normalized coordinates (0-1)
3. POINT prompts: Click points [[x, y], ...] with labels [1 for positive, 0 for negative]

For each user query:
1. Analyze the image carefully
2. Determine the best SAM3 prompt strategy
3. Generate the appropriate prompt(s)
4. Explain your reasoning

Respond in JSON format:
{
  "reasoning": "explanation of your analysis",
  "prompts": [
    {
      "type": "text" | "box" | "point",
      "text": "description" (for text prompts),
      "box": [cx, cy, w, h] (for box prompts),
      "points": [[x, y], ...] (for point prompts),
      "point_labels": [1, 0, ...] (for point prompts)
    }
  ]
}

Remember:
- For complex spatial queries ("leftmost", "behind", "next to"), analyze positions carefully
- For attribute-based queries ("wearing blue", "with stripes"), focus on visual features
- Prefer TEXT prompts for simple objects, BOX prompts when you can localize precisely
- You can generate multiple prompts for multiple objects
"""
    
    def __init__(
        self,
        sam3_url: str = "http://localhost:8000",
        qwen_url: str = "http://localhost:8001",
        sam3_api_version: str = "v1",
        qwen_api_version: str = "v1",
        debug: bool = False
    ):
        self.sam3_url = sam3_url.rstrip('/')
        self.qwen_url = qwen_url.rstrip('/')
        self.sam3_api_version = sam3_api_version
        self.qwen_api_version = qwen_api_version
        self.debug = debug
        
    def _image_to_base64(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """Конвертация изображения в base64"""
        if isinstance(image, str):
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(image, np.ndarray):
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
            return (image.shape[1], image.shape[0])
        else:
            raise ValueError("Unsupported image type")
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """Извлечь JSON из текстового ответа MLLM"""
        # Ищем JSON в тексте
        # Сначала пробуем найти между ```json и ```
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Пробуем найти просто JSON объект
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _normalize_bbox_from_llm(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Нормализует bbox от LLM в формат [cx, cy, w, h] (0-1)
        
        LLM может вернуть:
        - Абсолютные координаты [x1, y1, x2, y2]
        - Нормализованные координаты [x1, y1, x2, y2]
        - Уже центрированные [cx, cy, w, h]
        """
        if len(bbox) != 4:
            raise ValueError(f"Bbox должен содержать 4 значения, получено {len(bbox)}")
        
        # Проверяем, уже ли нормализованы значения (все в диапазоне 0-1)
        if all(0 <= v <= 1 for v in bbox):
            # Определяем формат
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                # Это [cx, cy, w, h]
                return bbox
            else:
                # Это [x1, y1, x2, y2], конвертируем в [cx, cy, w, h]
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                return [cx, cy, w, h]
        else:
            # Абсолютные координаты, нормализуем
            # Предполагаем формат [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            x1_norm = x1 / img_width
            y1_norm = y1 / img_height
            x2_norm = x2 / img_width
            y2_norm = y2 / img_height
            
            cx = (x1_norm + x2_norm) / 2
            cy = (y1_norm + y2_norm) / 2
            w = x2_norm - x1_norm
            h = y2_norm - y1_norm
            
            return [cx, cy, w, h]
    
    def generate_sam3_prompts(
        self,
        image: Union[str, Image.Image, np.ndarray],
        query: str
    ) -> Tuple[List[SAM3Prompt], str]:
        """
        Использует Qwen для анализа изображения и генерации промптов для SAM3
        
        Args:
            image: Изображение для анализа
            query: Запрос пользователя (например, "the leftmost child wearing blue vest")
            
        Returns:
            Tuple[List[SAM3Prompt], str]: Список промптов для SAM3 и reasoning от MLLM
        """
        image_b64 = self._image_to_base64(image)
        img_width, img_height = self._get_image_size(image)
        
        # Формируем промпт для MLLM
        user_prompt = f"""Analyze this image and generate SAM3 prompts to segment: "{query}"

Please provide your response in JSON format with reasoning and prompts."""
        
        print(f"🤖 Запрашиваем анализ у Qwen VLM...")
        if self.debug:
            print(f"   Query: {query}")
        
        # Вызываем Qwen image description с нашим системным промптом
        payload = {
            "image_base64": image_b64,
            "prompt": user_prompt,
            "detail_level": "comprehensive",
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        url = f"{self.qwen_url}/{self.qwen_api_version}/image/description"
        
        try:
            response = requests.post(url, json=payload, timeout=90)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Не удалось подключиться к Qwen API по адресу {url}"
            ) from e
        
        result = response.json()
        
        # Извлекаем ответ
        llm_response = result.get('result') or result.get('data') or {}
        if isinstance(llm_response, dict):
            llm_text = llm_response.get('description') or llm_response.get('text') or str(llm_response)
        else:
            llm_text = str(llm_response)
        
        print(f"📝 Ответ от Qwen:")
        print(f"{llm_text[:500]}...\n" if len(llm_text) > 500 else f"{llm_text}\n")
        
        # Пытаемся извлечь JSON из ответа
        parsed_json = self._extract_json_from_text(llm_text)
        
        if parsed_json and 'prompts' in parsed_json:
            # MLLM вернул структурированный ответ
            reasoning = parsed_json.get('reasoning', 'No reasoning provided')
            prompts_data = parsed_json.get('prompts', [])
            
            prompts = []
            for p in prompts_data:
                # Нормализуем bbox если он есть
                box = p.get('box')
                if box:
                    try:
                        box = self._normalize_bbox_from_llm(box, img_width, img_height)
                    except ValueError as e:
                        print(f"⚠️  Ошибка нормализации bbox: {e}")
                        box = None
                
                prompt = SAM3Prompt(
                    type=p.get('type', 'text'),
                    text=p.get('text'),
                    box=box,
                    points=p.get('points'),
                    point_labels=p.get('point_labels'),
                    reasoning=reasoning
                )
                prompts.append(prompt)
            
            return prompts, reasoning
        else:
            # MLLM не вернул JSON, используем fallback
            print("⚠️  MLLM не вернул структурированный JSON, используем текстовый промпт")
            
            # Создаем текстовый промпт на основе ответа
            prompt = SAM3Prompt(
                type="text",
                text=query,  # Используем исходный запрос
                reasoning=f"Fallback: using original query as text prompt. LLM response: {llm_text[:200]}"
            )
            
            return [prompt], llm_text
    
    def segment_with_sam3(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompts: List[SAM3Prompt],
        confidence_threshold: float = 0.5
    ) -> List[SegmentationResult]:
        """
        Сегментация с SAM3 используя промпты от MLLM
        
        Args:
            image: Изображение
            prompts: Список промптов от MLLM
            confidence_threshold: Порог уверенности
            
        Returns:
            Список результатов сегментации
        """
        image_b64 = self._image_to_base64(image)
        
        # Конвертируем SAM3Prompts в формат API
        sam3_prompts = []
        for p in prompts:
            if p.type == "text" and p.text:
                sam3_prompts.append({"type": "text", "text": p.text})
            elif p.type == "box" and p.box:
                sam3_prompts.append({"type": "box", "box": p.box, "label": True})
            elif p.type == "point" and p.points:
                sam3_prompts.append({
                    "type": "point",
                    "points": p.points,
                    "point_labels": p.point_labels or [1] * len(p.points)
                })
        
        if not sam3_prompts:
            print("⚠️  Нет валидных промптов для SAM3")
            return []
        
        payload = {
            "image": image_b64,
            "prompts": sam3_prompts,
            "confidence_threshold": confidence_threshold
        }
        
        url = f"{self.sam3_url}/api/{self.sam3_api_version}/image/segment"
        
        print(f"🎯 Отправка промптов в SAM3...")
        print(f"   Промптов: {len(sam3_prompts)}")
        for i, p in enumerate(sam3_prompts):
            print(f"   [{i+1}] type={p['type']}, ", end="")
            if 'text' in p:
                print(f"text='{p['text']}'")
            elif 'box' in p:
                print(f"box={p['box']}")
            elif 'points' in p:
                print(f"points={len(p['points'])} points")
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Не удалось подключиться к SAM3 API по адресу {url}"
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
                score=result['scores'][i],
                query="",  # Заполним позже
                prompt_used=prompts[i] if i < len(prompts) else None
            )
            segmentations.append(seg)
        
        return segmentations
    
    def segment(
        self,
        image: Union[str, Image.Image, np.ndarray],
        query: str,
        confidence_threshold: float = 0.5
    ) -> List[SegmentationResult]:
        """
        Полный пайплайн SAM3 Agent:
        1. MLLM анализирует изображение и запрос
        2. MLLM генерирует промпты для SAM3
        3. SAM3 сегментирует на основе промптов
        
        Args:
            image: Изображение для анализа
            query: Запрос пользователя (например, "the leftmost child wearing blue vest")
            confidence_threshold: Порог уверенности для SAM3
            
        Returns:
            Список результатов сегментации с контекстом от MLLM
        """
        print("=" * 70)
        print(f"🤖 SAM3 Agent - Анализ и сегментация")
        print(f"   Query: {query}")
        print("=" * 70)
        
        # Шаг 1: MLLM генерирует промпты
        prompts, reasoning = self.generate_sam3_prompts(image, query)
        
        print(f"\n💡 Reasoning от MLLM:")
        print(f"{reasoning}\n")
        
        if not prompts:
            print("⚠️  MLLM не сгенерировал промптов")
            return []
        
        # Шаг 2: SAM3 сегментирует
        segmentations = self.segment_with_sam3(
            image=image,
            prompts=prompts,
            confidence_threshold=confidence_threshold
        )
        
        # Добавляем контекст
        for seg in segmentations:
            seg.query = query
            seg.llm_reasoning = reasoning
        
        print(f"\n✅ Сегментировано объектов: {len(segmentations)}")
        for i, seg in enumerate(segmentations):
            print(f"   [{i+1}] score={seg.score:.2f}, bbox={seg.bbox.to_list()}")
            if seg.prompt_used:
                print(f"        prompt_type={seg.prompt_used.type}")
        
        print("=" * 70)
        
        return segmentations
    
    def visualize_results(
        self,
        image: Union[str, Image.Image],
        results: List[SegmentationResult],
        output_path: str = "output.jpg"
    ):
        """
        Визуализация результатов сегментации
        
        Args:
            image: Исходное изображение
            results: Результаты сегментации
            output_path: Путь для сохранения визуализации
        """
        # Загружаем изображение
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = image.copy().convert('RGB')
        
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Рисуем bbox для каждого результата
        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
        
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            
            # Конвертируем нормализованный bbox в абсолютные координаты
            x1, y1, x2, y2 = result.bbox.to_xyxy(width, height)
            
            # Убеждаемся что координаты в пределах изображения
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            if self.debug:
                print(f"   Визуализация объекта {i+1}:")
                print(f"     Нормализованный bbox: {result.bbox.to_list()}")
                print(f"     Абсолютный bbox: ({x1}, {y1}, {x2}, {y2})")
                print(f"     Размер изображения: {width}x{height}")
            
            # Рисуем bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
            # Добавляем текст с фоном для лучшей читаемости
            text = f"#{i+1}: {result.score:.2f}"
            
            # Пытаемся загрузить шрифт, если не получается - используем дефолтный
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Получаем размер текста
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Рисуем фон для текста
            text_bg_y = max(0, y1 - text_height - 8)
            draw.rectangle(
                [x1, text_bg_y, x1 + text_width + 8, text_bg_y + text_height + 8],
                fill=color
            )
            
            # Рисуем текст
            draw.text((x1 + 4, text_bg_y + 4), text, fill='white', font=font)
        
        # Сохраняем
        img.save(output_path)
        print(f"💾 Визуализация сохранена: {output_path}")
        print(f"   Размер: {width}x{height}")
        print(f"   Объектов: {len(results)}")
        
        return output_path


# ==================== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ====================

def example_complex_query():
    """Пример с комплексным запросом (как в sam3_agent.ipynb)"""
    print("=" * 60)
    print("ПРИМЕР: Комплексный запрос к SAM3 Agent")
    print("=" * 60)
    
    agent = SAM3QwenAgent(
        sam3_url="http://localhost:8000",
        qwen_url="http://localhost:8001",
        debug=True
    )
    
    # Примеры комплексных запросов
    test_cases = [
        {
            "image": "/home/golovanks/projects/sgp_kras/MainHanlder/CT/anno_agent-main/tmp_cvat_download/images/00001.jpg",
            "query": "the largest container in the image"
        },
        {
            "image": "/home/golovanks/projects/sgp_kras/MainHanlder/CT/anno_agent-main/tmp_cvat_download/images/00001.jpg",
            "query": "all containers"
        },
        {
            "image": "/home/golovanks/projects/sgp_kras/MainHanlder/CT/anno_agent-main/tmp_cvat_download/images/00001.jpg",
            "query": "the container on the left side"
        }
    ]
    
    for i, test in enumerate(test_cases[:1]):  # Запускаем только первый для примера
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}")
        print(f"{'='*60}")
        
        try:
            results = agent.segment(
                image=test["image"],
                query=test["query"],
                confidence_threshold=0.5
            )
            
            if results:
                # Визуализируем
                output_path = f"output_test_{i+1}.jpg"
                agent.visualize_results(test["image"], results, output_path)
                
                print(f"\n📊 Результаты для '{test['query']}':")
                for j, result in enumerate(results):
                    print(f"\n  Объект {j+1}:")
                    print(f"    Score: {result.score:.3f}")
                    print(f"    BBox: {result.bbox.to_list()}")
                    if result.prompt_used:
                        print(f"    Prompt type: {result.prompt_used.type}")
                        if result.prompt_used.text:
                            print(f"    Prompt text: {result.prompt_used.text}")
            else:
                print("  ⚠️  Объекты не найдены")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()


def example_simple():
    """Простой пример"""
    print("=" * 60)
    print("ПРИМЕР: Простой запрос")
    print("=" * 60)
    
    agent = SAM3QwenAgent()
    
    try:
        results = agent.segment(
            image="/home/golovanks/projects/sgp_kras/MainHanlder/CT/anno_agent-main/tmp_cvat_download/images/00001.jpg",
            query="container",
            confidence_threshold=0.5
        )
        
        if results:
            print(f"\n✅ Найдено объектов: {len(results)}")
            agent.visualize_results(
                "/home/golovanks/projects/sgp_kras/MainHanlder/CT/anno_agent-main/tmp_cvat_download/images/00001.jpg",
                results,
                "simple_output.jpg"
            )
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    """
    SAM3 Agent - MLLM использует SAM3 как инструмент
    
    Перед запуском убедитесь, что:
    1. SAM3 API запущен на http://localhost:8000
    2. Qwen VLM API запущен на http://localhost:8001
    """
    
    # Запускаем пример с комплексным запросом
    example_complex_query()
    
    # Или простой пример
    # example_simple()

# 🚀 Быстрый старт

## Установка за 5 минут

```bash
# 1. Клонирование и установка
git clone <repo_url>
cd ai_annotation_agent
pip install poetry
poetry install

# 2. Настройка
cp .env.example .env
# Отредактируйте .env файл

# 3. Проверка установки
poetry run python -c "from src.agents.video_annotation_agent import VideoAnnotationAgent; print('OK')"
```

## Примеры использования через CLI

### 1️⃣ Базовая разметка видео

```bash
poetry run python -m src.utils.cli video-basic \
  data/input/video.mp4 \
  --task "Detect all vehicles and pedestrians" \
  --max-frames 50 \
  --output output/my_task
```

**Что происходит:**
- Извлекается до 50 кадров из видео
- Каждый кадр анализируется моделью Qwen3-VL
- Создаются JSON файлы с аннотациями
- Результаты сохраняются в `output/my_task_TIMESTAMP/`

### 2️⃣ Разметка видео с опорными кадрами

```bash
# Сначала подготовьте опорные кадры с аннотациями в data/keyframes/
# Файлы должны быть: frame_001.jpg, frame_001_annotation.json, и т.д.

poetry run python -m src.utils.cli video-keyframes \
  data/input/video.mp4 \
  data/keyframes/ \
  --task "Track vehicles maintaining consistent IDs" \
  --max-frames 100
```

**Преимущества:**
- Консистентность разметки по всему видео
- Сохранение tracking ID объектов
- Более точная разметка благодаря референсным примерам

### 3️⃣ Разметка изображений с примерами

```bash
# Подготовьте примеры в data/examples/:
# - image1.jpg, image1_annotation.json
# - image2.jpg, image2_annotation.json

poetry run python -m src.utils.cli images-examples \
  data/images/unlabeled/ \
  data/examples/ \
  --task "Annotate products for e-commerce catalog"
```

**Few-shot learning:**
- Модель учится на ваших примерах
- Воспроизводит стиль разметки
- Подходит для специфических задач

### 4️⃣ Базовая разметка изображений

```bash
poetry run python -m src.utils.cli images-basic \
  data/images/raw/ \
  --task "Detect objects in street scenes"
```

**Простейший вариант:**
- Не требует примеров
- Работает по текстовому описанию
- Быстрый старт

## Программное использование

### Пример 1: Скрипт для разметки одного видео

```python
# annotate_video.py
import asyncio
from pathlib import Path
from src.agents.video_annotation_agent import VideoAnnotationAgent

async def main():
    agent = VideoAnnotationAgent()
    
    result = await agent.annotate_video_basic(
        video_path=Path("input.mp4"),
        task_description="""
        Detect:
        - All vehicles (cars, trucks, buses)
        - Pedestrians
        - Traffic signs
        Provide bounding boxes and confidence scores.
        """,
        max_frames=50
    )
    
    print(f"✅ Done! Check: {result['output_dir']}")
    await agent.cleanup()

asyncio.run(main())
```

Запуск:
```bash
poetry run python annotate_video.py
```

### Пример 2: Batch обработка множества файлов

```python
# batch_annotate.py
import asyncio
from pathlib import Path
from src.utils.batch_processor import BatchProcessor
from src.agents.image_annotation_agent import ImageAnnotationAgent

async def main():
    processor = BatchProcessor(max_concurrent=3)
    agent = ImageAnnotationAgent()
    
    # Найти все папки с изображениями
    input_dirs = list(Path("data/batches").glob("batch_*"))
    
    # Подготовить задачи
    tasks = []
    for dir_path in input_dirs:
        tasks.append((
            (),  # args
            {    # kwargs
                "images_dir": dir_path,
                "task_description": "Standard object detection",
                "output_dir": Path(f"output/{dir_path.name}")
            }
        ))
    
    # Запустить параллельную обработку
    results = await processor.process_batch(
        tasks,
        agent.annotate_images_basic,
        desc="Batch annotation"
    )
    
    # Статистика
    successful = sum(1 for r in results if r["success"])
    print(f"✅ Processed {successful}/{len(results)} batches")
    
    await agent.cleanup()

asyncio.run(main())
```

Запуск:
```bash
poetry run python batch_annotate.py
```

### Пример 3: Интеграция в pipeline

```python
# pipeline.py
import asyncio
from pathlib import Path
from src.agents.video_annotation_agent import VideoAnnotationAgent
from src.tools.validation_tools.py import AnnotationValidator

async def video_annotation_pipeline(video_path: Path):
    """Полный pipeline: экстракция → аннотация → валидация"""
    
    # 1. Аннотация
    agent = VideoAnnotationAgent()
    result = await agent.annotate_video_basic(
        video_path=video_path,
        task_description="Detect objects",
        max_frames=100
    )
    await agent.cleanup()
    
    # 2. Валидация
    validator = AnnotationValidator()
    annotations_dir = Path(result['output_dir']) / 'annotations'
    
    validation_results = []
    for ann_file in annotations_dir.glob("*.json"):
        validation = validator.validate_annotation_file(ann_file)
        validation_results.append(validation)
    
    # 3. Отчет
    valid_count = sum(1 for v in validation_results if v["valid"])
    print(f"Validation: {valid_count}/{len(validation_results)} valid")
    
    # Показать ошибки
    for v in validation_results:
        if not v["valid"]:
            print(f"❌ {v['file']}: {v['errors']}")
    
    return result

# Запуск
asyncio.run(video_annotation_pipeline(Path("video.mp4")))
```

## Структура данных

### Входные данные

```
data/
├── input/
│   ├── video1.mp4
│   └── video2.mp4
├── images/
│   ├── unlabeled/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── examples/
│       ├── example1.jpg
│       ├── example1_annotation.json
│       ├── example2.jpg
│       └── example2_annotation.json
└── keyframes/
    ├── keyframe_001.jpg
    ├── keyframe_001_annotation.json
    ├── keyframe_050.jpg
    └── keyframe_050_annotation.json
```

### Выходные данные

```
output/
└── video_basic_20260114_153045/
    ├── frames/
    │   ├── frame_000001.jpg
    │   ├── frame_000002.jpg
    │   └── ...
    ├── annotations/
    │   ├── frame_000001.json
    │   ├── frame_000002.json
    │   └── ...
    └── metadata/
        └── task_metadata.json
```

### Формат аннотации (JSON)

```json
{
  "annotations": [
    {
      "object": "car",
      "bbox": [100, 150, 300, 400],
      "confidence": 0.95,
      "attributes": {
        "color": "red",
        "type": "sedan",
        "tracking_id": "vehicle_001"
      }
    },
    {
      "object": "person",
      "bbox": [450, 200, 550, 500],
      "confidence": 0.89,
      "attributes": {
        "activity": "walking",
        "clothing": "casual"
      }
    }
  ],
  "metadata": {
    "image_analysis": "Street scene with light traffic",
    "frame_number": 42,
    "timestamp": "2026-01-14T15:30:45",
    "weather": "clear",
    "lighting": "daylight"
  }
}
```

## Настройка промптов

### Кастомный промпт для специфической задачи

```python
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["task_description", "frame_number"],
    template="""
You are a specialized medical image annotator.

Task: {task_description}
Frame: {frame_number}

Requirements:
1. Identify all anatomical structures
2. Mark any anomalies or pathologies
3. Provide confidence scores
4. Use medical terminology
5. Include measurements if applicable

Output format: JSON with structure:
{{
  "anatomical_structures": [...],
  "anomalies": [...],
  "measurements": {{}},
  "clinical_notes": "..."
}}
"""
)

# Использование
from src.agents.video_annotation_agent import VideoAnnotationAgent

agent = VideoAnnotationAgent()
# Установите кастомный промпт через monkey patching или модификацию класса
```

## Мониторинг и отладка

### Проверка логов

```bash
# Логи в реальном времени
tail -f logs/app_$(date +%Y-%m-%d).log

# Поиск ошибок
grep ERROR logs/app_*.log

# Статистика обработки
grep "Annotating" logs/app_*.log | wc -l
```

### Проверка качества аннотаций

```python
# validate_all.py
from pathlib import Path
from src.tools.validation_tools import AnnotationValidator
import json

validator = AnnotationValidator()
annotations_dir = Path("output/my_task_20260114_153045/annotations")

all_valid = True
for ann_file in annotations_dir.glob("*.json"):
    result = validator.validate_annotation_file(ann_file)
    
    if not result["valid"]:
        print(f"❌ {ann_file.name}")
        for error in result["errors"]:
            print(f"  - {error}")
        all_valid = False

if all_valid:
    print("✅ All annotations are valid!")
```

## Troubleshooting

### Проблема: "Connection refused" к Qwen API

```bash
# Проверьте, что API запущен
curl http://localhost:8000/health

# Проверьте .env
cat .env | grep QWEN_API_URL
```

### Проблема: Слишком медленная обработка

```python
# Решение: используйте batch processing с ограничением конкурентности
from src.utils.batch_processor import BatchProcessor

processor = BatchProcessor(max_concurrent=5)  # Увеличьте число
```

### Проблема: Некачественные аннотации

```python
# Решение 1: Улучшите промпт
task_description = """
Be very specific and detailed.
For each object provide:
- Exact bounding box coordinates
- High confidence (>0.8) detections only
- Detailed attributes
"""

# Решение 2: Используйте few-shot learning с примерами
# См. Пример 3
```

## Следующие шаги

1. ✅ Установите и протестируйте базовый пример
2. 📝 Адаптируйте промпты под вашу задачу
3. 🎯 Подготовьте примеры аннотаций для few-shot learning
4. 🚀 Запустите batch обработку на полном датасете
5. ✔️ Валидируйте результаты
6. 🔄 Итеративно улучшайте промпты

## Полезные команды

```bash
# Тестирование
poetry run pytest

# Форматирование кода
poetry run black src/
poetry run ruff check src/

# Проверка типов
poetry run mypy src/

# Генерация документации
poetry run pdoc src/ --html --output-dir docs/
```
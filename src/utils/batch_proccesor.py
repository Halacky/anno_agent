
import asyncio
from typing import List, Dict, Callable, Any
from pathlib import Path
from loguru import logger
from tqdm import tqdm

class BatchProcessor:
    """Параллельная обработка задач с контролем конкурентности"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_task(self, task_func: Callable, *args, **kwargs) -> Any:
        """Обработка одной задачи с семафором"""
        async with self.semaphore:
            try:
                result = await task_func(*args, **kwargs)
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Task failed: {e}")
                return {"success": False, "error": str(e)}
    
    async def process_batch(
        self,
        tasks: List[tuple],
        task_func: Callable,
        desc: str = "Processing"
    ) -> List[Dict]:
        """
        Обработка батча задач
        
        Args:
            tasks: Список кортежей (args, kwargs) для каждой задачи
            task_func: Асинхронная функция для обработки
            desc: Описание для progress bar
        """
        logger.info(f"Starting batch processing: {len(tasks)} tasks")
        
        # Создание корутин для всех задач
        coroutines = []
        for args, kwargs in tasks:
            coro = self.process_task(task_func, *args, **kwargs)
            coroutines.append(coro)
        
        # Выполнение с progress bar
        results = []
        with tqdm(total=len(coroutines), desc=desc) as pbar:
            for coro in asyncio.as_completed(coroutines):
                result = await coro
                results.append(result)
                pbar.update(1)
        
        # Подсчет статистики
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        logger.info(f"Batch completed: {successful} successful, {failed} failed")
        
        return results
    
"""
Пример использования batch processor:

from src.utils.batch_processor import BatchProcessor
from src.agents.image_annotation_agent import ImageAnnotationAgent

async def batch_annotate_directories():
    processor = BatchProcessor(max_concurrent=3)
    agent = ImageAnnotationAgent()
    
    # Список директорий для обработки
    directories = [
        Path("data/batch1"),
        Path("data/batch2"),
        Path("data/batch3")
    ]
    
    # Подготовка задач
    tasks = []
    for dir_path in directories:
        args = ()
        kwargs = {
            "images_dir": dir_path,
            "task_description": "Detect objects",
            "output_dir": Path(f"output/{dir_path.name}")
        }
        tasks.append((args, kwargs))
    
    # Запуск batch обработки
    results = await processor.process_batch(
        tasks,
        agent.annotate_images_basic,
        desc="Annotating directories"
    )
    
    await agent.cleanup()
    return results

# Запуск
# asyncio.run(batch_annotate_directories())
"""

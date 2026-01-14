"""
tests/test_agents.py - Примеры юнит-тестов для агентов
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from src.agents.video_annotation_agent import VideoAnnotationAgent
from src.agents.image_annotation_agent import ImageAnnotationAgent
from src.core.qwen_client import QwenVLClient

@pytest.fixture
def mock_qwen_client():
    """Мок клиента Qwen для тестирования без реального API"""
    client = Mock(spec=QwenVLClient)
    client.analyze_image = AsyncMock(return_value={
        "success": True,
        "content": '''
        {
            "annotations": [
                {
                    "object": "test_object",
                    "bbox": [0, 0, 100, 100],
                    "confidence": 0.95,
                    "attributes": {}
                }
            ],
            "metadata": {"test": true}
        }
        '''
    })
    client.close = AsyncMock()
    return client

@pytest.fixture
def sample_video_path(tmp_path):
    """Создание тестового видео файла"""
    import cv2
    import numpy as np
    
    video_path = tmp_path / "test_video.mp4"
    
    # Создание простого тестового видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    for i in range(30):  # 1 секунда видео
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    return video_path

@pytest.fixture
def sample_images(tmp_path):
    """Создание тестовых изображений"""
    import cv2
    import numpy as np
    
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    
    for i in range(3):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"test_{i}.jpg"), img)
    
    return images_dir

class TestVideoAnnotationAgent:
    @pytest.mark.asyncio
    async def test_annotate_video_basic(self, mock_qwen_client, sample_video_path, tmp_path):
        """Тест базовой разметки видео"""
        agent = VideoAnnotationAgent(qwen_client=mock_qwen_client)
        
        result = await agent.annotate_video_basic(
            video_path=sample_video_path,
            task_description="Test annotation",
            output_dir=tmp_path,
            max_frames=5
        )
        
        assert result["status"] == "success"
        assert "metadata" in result
        assert "annotations" in result
        assert result["metadata"]["total_frames"] > 0
        
        await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_annotate_with_keyframes(self, mock_qwen_client, sample_video_path, tmp_path):
        """Тест разметки с опорными кадрами"""
        agent = VideoAnnotationAgent(qwen_client=mock_qwen_client)
        
        # Подготовка mock опорных кадров
        keyframe_annotations = {
            Path("frame_1.jpg"): {
                "annotations": [{"object": "test", "bbox": [0, 0, 10, 10]}]
            }
        }
        
        result = await agent.annotate_video_with_keyframes(
            video_path=sample_video_path,
            task_description="Test with keyframes",
            keyframe_annotations=keyframe_annotations,
            output_dir=tmp_path,
            max_frames=5
        )
        
        assert result["status"] == "success"
        assert result["metadata"]["keyframes_used"] == 1
        
        await agent.cleanup()

class TestImageAnnotationAgent:
    @pytest.mark.asyncio
    async def test_annotate_images_basic(self, mock_qwen_client, sample_images, tmp_path):
        """Тест базовой разметки изображений"""
        agent = ImageAnnotationAgent(qwen_client=mock_qwen_client)
        
        result = await agent.annotate_images_basic(
            images_dir=sample_images,
            task_description="Test annotation",
            output_dir=tmp_path
        )
        
        assert result["status"] == "success"
        assert result["metadata"]["total_images"] == 3
        assert len(result["annotations"]) > 0
        
        await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_annotate_with_examples(self, mock_qwen_client, sample_images, tmp_path):
        """Тест разметки с примерами"""
        agent = ImageAnnotationAgent(qwen_client=mock_qwen_client)
        
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        
        # Создание примера
        import shutil
        example_img = list(sample_images.glob("*.jpg"))[0]
        shutil.copy(example_img, examples_dir / "example.jpg")
        
        result = await agent.annotate_images_with_examples(
            images_dir=sample_images,
            examples_dir=examples_dir,
            task_description="Test with examples",
            output_dir=tmp_path
        )
        
        assert result["status"] == "success"
        assert "examples_used" in result["metadata"]
        
        await agent.cleanup()

class TestQwenClient:
    @pytest.mark.asyncio
    async def test_analyze_image_success(self, tmp_path):
        """Тест успешного анализа изображения"""
        import cv2
        import numpy as np
        
        # Создание тестового изображения
        img_path = tmp_path / "test.jpg"
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        
        # Mock HTTP клиента
        with patch('src.core.qwen_client.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"annotations": []}'
                    }
                }],
                "usage": {}
            }
            mock_response.raise_for_status = Mock()
            
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.aclose = AsyncMock()
            
            client = QwenVLClient()
            result = await client.analyze_image(img_path, "Test prompt")
            
            assert result["success"] == True
            assert "content" in result
            
            await client.close()
    
    @pytest.mark.asyncio
    async def test_analyze_image_error_handling(self):
        """Тест обработки ошибок при анализе"""
        with patch('src.core.qwen_client.httpx.AsyncClient') as mock_client:
            mock_client.return_value.post = AsyncMock(side_effect=Exception("API Error"))
            mock_client.return_value.aclose = AsyncMock()
            
            client = QwenVLClient()
            result = await client.analyze_image(Path("nonexistent.jpg"), "Test")
            
            assert result["success"] == False
            assert "error" in result
            
            await client.close()

"""
tests/test_tools.py - Тесты для инструментов
"""

import pytest
from pathlib import Path
import cv2
import numpy as np
from src.core.video_processor import VideoProcessor

class TestVideoProcessor:
    @pytest.fixture
    def test_video(self, tmp_path):
        """Создание тестового видео"""
        video_path = tmp_path / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        
        for i in range(60):  # 2 секунды
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Добавляем изменение цвета для детекции сцен
            if i < 30:
                frame[:, :] = [0, 0, 255]  # Красный
            else:
                frame[:, :] = [255, 0, 0]  # Синий
            out.write(frame)
        
        out.release()
        return video_path
    
    def test_extract_frames(self, test_video, tmp_path):
        """Тест извлечения кадров"""
        output_dir = tmp_path / "frames"
        output_dir.mkdir()
        
        processor = VideoProcessor()
        frames = processor.extract_frames(
            test_video,
            output_dir,
            max_frames=10
        )
        
        assert len(frames) == 10
        assert all(f.exists() for f in frames)
    
    def test_detect_scene_changes(self, test_video, tmp_path):
        """Тест детекции изменения сцен"""
        output_dir = tmp_path / "keyframes"
        output_dir.mkdir()
        
        processor = VideoProcessor()
        keyframes = processor.detect_scene_changes(
            test_video,
            output_dir,
            threshold=50.0
        )
        
        # Должна быть обнаружена смена сцены
        assert len(keyframes) > 0

"""
tests/conftest.py - Общие фикстуры для тестов
"""

import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="session")
def test_data_dir():
    """Временная директория для тестовых данных"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def clean_output_dir(tmp_path):
    """Чистая выходная директория для каждого теста"""
    output = tmp_path / "output"
    output.mkdir()
    return output
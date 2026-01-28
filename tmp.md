Отлично! Сейчас изучу официальный репозиторий SAM2/SAM3 от Meta.Отлично! Теперь давайте посмотрим на официальные примеры работы с видео:Теперь давайте посмотрим на конкретный пример работы с видео:Отлично! Нашел ключевую информацию. Давайте посмотрим на исходный код `add_prompt`:Страница загрузилась некорректно. Попробую через raw ссылку:Отлично! Я нашел несколько критически важных issue. Теперь понятна проблема! Давайте посмотрим на Issue #263, которая **точно описывает вашу проблему**:

## ✅ **РЕШЕНИЕ НАЙДЕНО!**

Согласно [Issue #263](https://github.com/facebookresearch/sam3/issues/263) и [Issue #193](https://github.com/facebookresearch/sam3/issues/193), проблема в следующем:

### **Проблема:**

При использовании боксов с текстовым промптом "visual" (или любым другим) модель не детектирует объекты на остальных кадрах видео, кроме первого.

### **Корневая причина:**

SAM3 имеет **ДВА разных режима** работы с видео:

1. **PCS (Promptable Concept Segmentation)** - текстовые промпты для поиска ВСЕХ объектов класса
2. **PVS (Promptable Visual Segmentation)** - визуальные промпты (боксы/точки) для отслеживания КОНКРЕТНОГО объекта

**КРИТИЧНО**: Когда вы передаете и бокс, и текст одновременно, модель переходит в режим PCS и ИГНОРИРУЕТ боксы!

### **Решение для вашего случая:**

Вам нужно использовать **ТОЛЬКО боксы БЕЗ текстового промпта** И добавить **`obj_id`**:

```python
response = predictor.handle_request(
    request={
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "bounding_boxes": [[x, y, w, h]],  # ТОЛЬКО боксы
        "bounding_box_labels": [1],         # Positive prompt
        "obj_id": 1,                        # ОБЯЗАТЕЛЬНО!
        # "text": None  ← НЕ передавайте вообще!
    }
)
```

## Исправленный код для вашего случая:

```python
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from sam3.logger import get_logger
from sam3.model.sam3_video_predictor import (
    Sam3VideoPredictor,
    Sam3VideoPredictorMultiGPU,
)
from sam3.model_builder import build_sam3_video_model
from sam3.agent.helpers.rle import rle_encode

logger = get_logger(__name__)


class SAM3VideoModel:

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        bpe_path: Optional[str] = None,
        gpu_ids: Optional[List[int]] = None,
        video_loader_type: str = "cv2",
        async_loading_frames: bool = False,
    ):
        # ... ваш существующий код инициализации ...
        load_from_HF = False
        resolved_checkpoint_path = None

        if checkpoint is None:
            local_checkpoint = "/app/server/sam_weights/sam3.pt"
            if Path(local_checkpoint).exists():
                resolved_checkpoint_path = local_checkpoint
            else:
                load_from_HF = True
        else:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.is_file():
                resolved_checkpoint_path = checkpoint
            elif checkpoint_path.is_dir():
                checkpoint_file = checkpoint_path / "sam3.pt"
                if checkpoint_file.exists():
                    resolved_checkpoint_path = str(checkpoint_file)
                else:
                    raise FileNotFoundError(f"Checkpoint file 'sam3.pt' not found in directory: {checkpoint}")
            else:
                if "/" in checkpoint and not checkpoint.startswith(("/", ".")):
                    local_checkpoint = "/app/server/sam_weights/sam3.pt"
                    if Path(local_checkpoint).exists():
                        logger.info(f"Using local checkpoint instead of HuggingFace: {local_checkpoint}")
                        resolved_checkpoint_path = local_checkpoint
                    else:
                        load_from_HF = True
                else:
                    local_checkpoint = "/app/server/sam_weights/sam3.pt"
                    if Path(local_checkpoint).exists():
                        resolved_checkpoint_path = local_checkpoint
                    else:
                        load_from_HF = True

        logger.info(f"Using video checkpoint path: {resolved_checkpoint_path}, load_from_HF: {load_from_HF}")

        self.checkpoint = resolved_checkpoint_path
        self.bpe_path = bpe_path
        self.gpu_ids = gpu_ids or [0]
        self.video_loader_type = video_loader_type
        self.async_loading_frames = async_loading_frames

        available_gpus = torch.cuda.device_count()
        filtered_gpu_ids = [gpu_id for gpu_id in self.gpu_ids if gpu_id < available_gpus]
        if not filtered_gpu_ids:
            filtered_gpu_ids = [0]

        self.gpu_ids = filtered_gpu_ids

        if len(self.gpu_ids) > 1:
            logger.info(f"Initializing multi-GPU predictor with GPUs: {self.gpu_ids}")
            self.predictor = Sam3VideoPredictorMultiGPU(
                checkpoint_path=resolved_checkpoint_path,
                bpe_path=bpe_path,
                gpus_to_use=self.gpu_ids,
                video_loader_type=video_loader_type,
                async_loading_frames=async_loading_frames,
            )
        else:
            logger.info(f"Initializing single-GPU predictor on GPU: {self.gpu_ids[0]}")
            torch.cuda.set_device(self.gpu_ids[0])
            self.predictor = Sam3VideoPredictor(
                checkpoint_path=resolved_checkpoint_path,
                bpe_path=bpe_path,
                video_loader_type=video_loader_type,
                async_loading_frames=async_loading_frames,
            )

        logger.info("SAM3 video model initialized successfully")

    def start_session(
        self, video_path: str, session_id: Optional[str] = None
    ) -> Tuple[str, Dict]:
        # ... ваш существующий код ...
        start_time = time.time()

        request = {
            "type": "start_session",
            "resource_path": video_path,
            "session_id": session_id,
        }

        response = self.predictor.handle_request(request)
        session_id = response["session_id"]

        session = self.predictor._get_session(session_id)
        inference_state = session["state"]

        try:
            num_frames = inference_state.get("num_frames", 0)
            input_batch = inference_state.get("input_batch")

            if input_batch is not None and hasattr(input_batch, "find_inputs"):
                for t in range(num_frames):
                    if t < len(input_batch.find_inputs):
                        find_input = input_batch.find_inputs[t]
                        if hasattr(find_input, "text_ids"):
                            text_ids = find_input.text_ids
                            if isinstance(text_ids, list):
                                find_input.text_ids = torch.tensor(
                                    text_ids, dtype=torch.long, device=self.predictor.device
                                )
                                logger.debug(f"Converted text_ids to tensor for frame {t}")
        except Exception as e:
            logger.warning(f"Could not patch text_ids in inference_state: {e}")

        width = 0
        height = 0

        if hasattr(inference_state, "get"):
            width = (
                inference_state.get("video_width")
                or inference_state.get("width")
                or inference_state.get("_video_width", 0)
            )
            height = (
                inference_state.get("video_height")
                or inference_state.get("height")
                or inference_state.get("_video_height", 0)
            )

        if width == 0 or height == 0:
            import cv2

            session_video_path = session.get("video_path", "")
            if session_video_path and session_video_path != "":
                cap = cv2.VideoCapture(session_video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
            else:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

        video_info = {
            "total_frames": inference_state.get("num_frames", 0),
            "resolution": {
                "width": width,
                "height": height,
            },
            "fps": 30.0,
            "duration_seconds": inference_state.get("num_frames", 0) / 30.0,
        }

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"Started session {session_id} for video with {video_info['total_frames']} frames "
            f"({video_info['resolution']['width']}x{video_info['resolution']['height']}) "
            f"in {elapsed:.1f}ms"
        )

        return session_id, video_info

    def add_prompt(
        self,
        session_id: str,
        frame_index: int,
        obj_id: int,  # ОБЯЗАТЕЛЬНЫЙ параметр
        boxes: Optional[List[List[float]]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        box_labels: Optional[List[int]] = None,
    ):
        """
        Добавляет ВИЗУАЛЬНЫЙ промпт (боксы/точки) для отслеживания конкретного объекта.
        
        КРИТИЧНО: НЕ используйте text_prompt с боксами! 
        Это переключает модель в режим PCS и боксы игнорируются.
        
        Для отслеживания конкретного объекта используйте ТОЛЬКО:
        - boxes (список координат [x, y, w, h])
        - points (список координат [[x, y]])
        - obj_id (уникальный ID для каждого отслеживаемого объекта)
        """
        start_time = time.time()

        if not isinstance(frame_index, int):
            logger.warning(f"frame_index is not int: {type(frame_index)} = {frame_index}")
            frame_index = int(frame_index) if hasattr(frame_index, "int") else frame_index[0]

        if boxes is None and points is None:
            raise ValueError("Either 'boxes' or 'points' must be provided")

        # КРИТИЧНО: НЕ передаем text!
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "obj_id": obj_id,
            "bounding_boxes": boxes,
            "bounding_box_labels": box_labels or [1] * len(boxes) if boxes else None,
            "points": points,
            "point_labels": point_labels,
            # НЕТ "text"!
        }

        logger.info(
            f"Adding VISUAL prompt: obj_id={obj_id}, frame={frame_index}, "
            f"boxes={boxes}, points={points}"
        )

        response = self.predictor.handle_request(request)
        logger.info("Handle request successful")

        frame_idx = response["frame_index"]
        outputs = response["outputs"]

        masks_rle = []
        boxes_xywh = []
        scores = []

        obj_ids = outputs["out_obj_ids"]
        probs = outputs["out_probs"]
        boxes_out = outputs["out_boxes_xywh"]
        masks = outputs["out_binary_masks"]

        for i in range(len(obj_ids)):
            mask_tensor = torch.from_numpy(masks[i]).unsqueeze(0)
            mask_rle = rle_encode(mask_tensor)
            masks_rle.append(mask_rle[0]["counts"])

            try:
                box_val = boxes_out[i].tolist() if hasattr(boxes_out[i], "tolist") else boxes_out[i]
                boxes_xywh.append(box_val)
            except:
                boxes_xywh.append(
                    boxes_out[i] if isinstance(boxes_out[i], list) else [float(boxes_out[i])]
                )

            scores.append(float(probs[i]))

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"Added prompt to frame {frame_idx} in session {session_id}: "
            f"obj_id={obj_id}, {len(masks_rle)} masks returned, {elapsed:.1f}ms"
        )

        return frame_idx, list(obj_ids), masks_rle, boxes_xywh, scores

    # ... остальные методы без изменений ...
    
    def propagate_in_video(
        self,
        session_id: str,
        direction: str = "both",
        start_frame_index: Optional[int] = None,
        max_frames: Optional[int] = None,
    ):
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": direction,
            "start_frame_index": start_frame_index,
            "max_frame_num_to_track": max_frames,
        }

        logger.info(f"Starting propagation: {request}")
        start_time = time.time()
        frames_processed = 0

        for response in self.predictor.handle_stream_request(request):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]

            obj_ids = outputs["out_obj_ids"]
            probs = outputs["out_probs"]
            boxes_xywh = outputs["out_boxes_xywh"]
            binary_masks = outputs["out_binary_masks"]

            objects = []
            for i in range(len(obj_ids)):
                obj_id = obj_ids[i]
                mask_np = binary_masks[i]
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).bool()

                mask_rle = rle_encode(mask_tensor)

                box = boxes_xywh[i].tolist()
                score = float(probs[i])

                objects.append(
                    {
                        "id": int(obj_id),
                        "mask": mask_rle[0]["counts"],
                        "box": box,
                        "score": score,
                    }
                )

            frames_processed += 1
            yield {"frame_index": frame_idx, "objects": objects}

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"Propagation completed for session {session_id}: "
            f"{frames_processed} frames in {elapsed:.1f}ms "
            f"({elapsed/frames_processed:.1f}ms/frame)"
        )

    def remove_object(self, session_id: str, obj_id: int) -> bool:
        request = {
            "type": "remove_object",
            "session_id": session_id,
            "obj_id": obj_id,
            "is_user_action": True,
        }

        response = self.predictor.handle_request(request)
        logger.info(f"Removed object {obj_id} from session {session_id}")
        return response["is_success"]

    def reset_session(self, session_id: str) -> bool:
        request = {"type": "reset_session", "session_id": session_id}

        response = self.predictor.handle_request(request)
        logger.info(f"Reset session {session_id}")
        return response["is_success"]

    def close_session(self, session_id: str) -> bool:
        request = {"type": "close_session", "session_id": session_id}

        response = self.predictor.handle_request(request)
        logger.info(f"Closed session {session_id}")
        return response["is_success"]

    def get_session_info(self, session_id: str) -> Dict:
        session = self.predictor._get_session(session_id)
        inference_state = session["state"]
        num_objects = len(inference_state.get("obj_ids", []))
        gpu_memory_mb = torch.cuda.memory_allocated() / (1024**2)

        return {
            "session_id": session_id,
            "num_frames": inference_state["num_frames"],
            "num_objects": num_objects,
            "gpu_memory_mb": gpu_memory_mb,
            "start_time": session["start_time"],
        }

    def list_sessions(self) -> List[str]:
        return list(self.predictor._ALL_INFERENCE_STATES.keys())

    def shutdown(self):
        self.predictor.shutdown()
        logger.info("SAM3 video model shutdown complete")
```

## Пример использования:

```python
# Инициализация
model = SAM3VideoModel()

# Создание сессии
session_id, video_info = model.start_session("video.mp4")

# ПРАВИЛЬНО: Добавление промптов каждые 10 кадров
# Используйте УНИКАЛЬНЫЙ obj_id для каждого объекта
container_id = 1

for frame_idx in [0, 10, 20, 30, 40]:  # Ключевые кадры
    # Ваш бокс для контейнера на этом кадре
    box = [100, 150, 200, 300]  # Замените на реальные координаты
    
    model.add_prompt(
        session_id=session_id,
        frame_index=frame_idx,
        obj_id=container_id,  # ОДИН И ТОТ ЖЕ ID для одного контейнера!
        boxes=[box],
        # НЕТ text_prompt!
    )

# Пропагация
for result in model.propagate_in_video(session_id):
    frame_idx = result["frame_index"]
    for obj in result["objects"]:
        if obj["id"] == container_id:
            print(f"Frame {frame_idx}: Container at {obj['box']}, score={obj['score']}")

model.close_session(session_id)
```

## Ключевые изменения:

1. ✅ **Удален `text_prompt`** из `add_prompt` - теперь используем ТОЛЬКО визуальные промпты
2. ✅ **`obj_id` стал обязательным** - для правильного отслеживания
3. ✅ **Добавлена документация** - объясняет почему нельзя смешивать text + boxes

Это должно решить вашу проблему! Теперь модель будет отслеживать **конкретный** контейнер, указанный боксом, а не все контейнеры на видео.

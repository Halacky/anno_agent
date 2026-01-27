import os
import io
import json
import base64
import zipfile
import shutil
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import requests
import tqdm
from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from pycocotools import mask as mask_utils

CVAT_HOST = "http://x.x.x.x:8080"
CVAT_USER = "GolovanKS"
CVAT_PASS = "CECigolik18"
SAM_API_ROOT = "https://x.x.x.x/samapi/v1/video"
SAM_GPU_IDS = [0]

TASK_NAME = "test_sam"
OUTPUT_ROOT = Path("./pipeline_output")

FPS = 8
VIDEO_CODEC = "mp4v"
TARGET_RESOLUTION = (1920, 1080)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

def _log(info: str):
    logging.info(info)


class CvatClient:
    def __init__(self, host: str, user: str, password: str):
        self.host = host
        self.user = user
        self.password = password
        self._client = None

    def __enter__(self):
        self._client = make_client(
            host=self.host,
            credentials=(self.user, self.password),
        )
        self._client.__enter__()
        _log(f"✅ Подключились к CVAT {self.host}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            self._client.__exit__(exc_type, exc_val, exc_tb)
        _log("🔌 Соединение с CVAT закрыто")

    def _find_task_by_name(self, name: str):
        for t in self._client.tasks.list():
            if t.name == name:
                return t
        raise RuntimeError(f"Задача с именем «{name}» не найдена")

    def download_images(self, task_name: str, output_dir: Path) -> List[Path]:
        task = self._find_task_by_name(task_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / f"tmp_{task.id}.zip"

        _log(f"📥 Экспортируем изображения задачи {task.id} → {zip_path}")
        task.export_dataset(
            format_name="YOLO 1.1",
            filename=str(zip_path),
            include_images=True,
        )

        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.infolist():
                if Path(member.filename).suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
                }:
                    name = Path(member.filename).name
                    target = images_dir / name
                    with z.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)

        zip_path.unlink()
        _log(f"✅ Скачано {len(list(images_dir.iterdir()))} изображений")
        return sorted(images_dir.iterdir())

    def download_yolo_annotations(self, task_name: str, output_dir: Path) -> Dict[str, List[str]]:
        task = self._find_task_by_name(task_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / f"anno_{task.id}.zip"
        task.export_dataset(
            format_name="YOLO 1.1",
            filename=str(zip_path),
            include_images=False,
        )

        ann_dict: Dict[str, List[str]] = {}
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.infolist():
                p = Path(member.filename)
                if p.suffix == ".txt" and p.name not in {"obj.names", "train.txt", "val.txt"}:
                    stem = p.stem
                    txt = z.read(member).decode("utf-8").strip()
                    ann_dict[stem] = txt.splitlines() if txt else []
        zip_path.unlink()
        _log(f"✅ Скачано {len(ann_dict)} файлов разметки (YOLO)")
        return ann_dict


def make_video_from_images(
    image_paths: List[Path],
    output_path: Path,
    fps: int = FPS,
    codec: str = VIDEO_CODEC,
) -> None:
    if not image_paths:
        raise RuntimeError("Список изображений пуст")

    first = cv2.imread(str(image_paths[0]))
    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    _log(f"▶️  Записываем видео {output_path} ({width}x{height} @ {fps}fps)")

    for p in tqdm.tqdm(image_paths, desc="Сборка видео"):
        frame = cv2.imread(str(p))
        if frame is None:
            raise RuntimeError(f"Не удалось прочитать {p}")
        writer.write(frame)

    writer.release()
    _log("✅ Видеофайл готов")


class SamClient:
    def __init__(self, api_root: str, gpu_ids: List[int] = None,
                 timeout: int = 120, verify_ssl: bool = False):
        self.api_root = api_root.rstrip("/")
        self.gpu_ids = gpu_ids or [0]
        self.session_id: str = ""
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    def start_session(self, video_path: Path) -> str:
        try:
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
            encoded_video_bytes = base64.b64encode(video_bytes)
            encoded_video_string = encoded_video_bytes.decode("utf-8")
        except FileNotFoundError:
            raise RuntimeError(f"Файл не найден: {video_path}")
        except Exception as e:
            raise RuntimeError(f"Ошибка при чтении видео: {e}")

        payload = {
            "video_base64": encoded_video_string,
            "gpu_ids": self.gpu_ids
        }
        response = requests.post(
            f"{self.api_root}/session/start",
            json=payload,
            verify=self.verify_ssl,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        self.session_id = data.get("session_id") or data.get("id")
        _log(f"✅ SAM‑сессия запущена, id={self.session_id}")
        return self.session_id

    def send_prompt(self, frame_index: int, prompts: List[dict]) -> dict:
        if not self.session_id:
            raise RuntimeError("Сессия ещё не создана")
        url = f"{self.api_root}/session/{self.session_id}/prompt"
        payload = {"frame_index": frame_index, "prompts": prompts}
        r = requests.post(url, json=payload, verify=self.verify_ssl, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def propagate(
        self,
        start_frame_index: int,
        direction: str = "both",
        max_frames: int = 0,
    ) -> dict:
        url = f"{self.api_root}/session/{self.session_id}/propagate"
        payload = {
            "direction": direction,
            "start_frame_index": start_frame_index,
            "max_frames": max_frames,
        }
        r = requests.post(url, json=payload, verify=self.verify_ssl, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_status(self) -> dict:
        """Получить полный статус сессии со всеми результатами"""
        url = f"{self.api_root}/session/{self.session_id}/status"
        r = requests.get(url, verify=self.verify_ssl, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_all_results(self) -> dict:
        """
        Получить все результаты сессии.
        Попробуем несколько возможных endpoint'ов.
        """
        # Вариант 1: /results
        try:
            url = f"{self.api_root}/session/{self.session_id}/results"
            r = requests.get(url, verify=self.verify_ssl, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except:
            pass
        
        # Вариант 2: /masks
        try:
            url = f"{self.api_root}/session/{self.session_id}/masks"
            r = requests.get(url, verify=self.verify_ssl, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except:
            pass
        
        # Вариант 3: через status (может содержать всё)
        return self.get_status()


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(min(v, hi), lo)


def yolo_to_prompts(
    yolo_lines: List[str],
    class_names: List[str],
) -> List[dict]:
    prompts: List[dict] = []

    for line in yolo_lines:
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id, cx, cy, w, h = map(float, parts)

        if w <= 0 or h <= 0:
            continue

        x_min = cx - w / 2.0
        y_min = cy - h / 2.0

        x_min = _clip(x_min)
        y_min = _clip(y_min)
        w = _clip(w)
        h = _clip(h)

        if x_min + w > 1.0 or y_min + h > 1.0:
            _log(f"Пропускаем некорректную аннотацию: {x_min},{y_min},{w},{h}")
            continue

        cls_name = (
            class_names[int(cls_id)]
            if 0 <= int(cls_id) < len(class_names)
            else f"class_{int(cls_id)}"
        )
        
        prompts.append({"type": "text", "text": cls_name})
        prompts.append({
            "type": "box",
            "box": [x_min, y_min, w, h],
            "label": True,
        })

    return prompts


def overlay_masks_on_frame(
    frame: np.ndarray,
    masks_data: List[dict],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Накладывает маски на кадр.
    """
    h, w = image_size
    overlay = frame.copy()

    for obj in masks_data:
        rle_data = None
        
        if isinstance(obj, dict):
            if 'mask' in obj:
                rle_data = obj['mask']
            elif 'rle' in obj:
                rle_data = obj['rle']
            elif 'counts' in obj:
                rle_data = obj
        elif isinstance(obj, str):
            rle_data = obj

        if rle_data is None:
            continue

        if isinstance(rle_data, str):
            rle = {"size": [h, w], "counts": rle_data.encode("utf-8")}
        elif isinstance(rle_data, dict):
            if 'counts' in rle_data and isinstance(rle_data['counts'], str):
                rle = {"size": [h, w], "counts": rle_data['counts'].encode("utf-8")}
            else:
                rle = rle_data
        else:
            continue

        try:
            binary_mask = mask_utils.decode(rle)
            if binary_mask.shape != (h, w):
                binary_mask = binary_mask.squeeze()

            color = np.random.randint(0, 255, size=3, dtype=np.uint8).tolist()
            overlay[binary_mask == 1] = color
        except Exception as e:
            _log(f"⚠️  Ошибка декодирования маски: {e}")
            continue

    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    return frame


def draw_boxes_on_frame(
    frame: np.ndarray,
    yolo_lines: List[str],
    class_names: List[str],
    thickness: int = 2,
) -> np.ndarray:
    height, width = frame.shape[:2]

    for line in yolo_lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, w, h = map(float, parts)
        x_min = int((cx - w / 2.0) * width)
        y_min = int((cy - h / 2.0) * height)
        x_max = int((cx + w / 2.0) * width)
        y_max = int((cy + h / 2.0) * height)

        cls_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"{cls_id}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(
            frame,
            cls_name,
            (x_min, max(y_min - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def run_pipeline():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Шаг 1: Скачиваем данные из CVAT
    with CvatClient(CVAT_HOST, CVAT_USER, CVAT_PASS) as cvat:
        images_dir = OUTPUT_ROOT / "cvat_images"
        yolo_ann = cvat.download_yolo_annotations(TASK_NAME, OUTPUT_ROOT / "cvat_ann")
        image_paths = cvat.download_images(TASK_NAME, OUTPUT_ROOT / "cvat_export")
        
    class_names = ["lift"]

    # Шаг 2: Создаём видео
    video_path = OUTPUT_ROOT / "source_video.mp4"
    make_video_from_images(image_paths, video_path, fps=FPS)

    # Шаг 3: Запускаем SAM сессию
    sam = SamClient(
        api_root=SAM_API_ROOT,
        gpu_ids=SAM_GPU_IDS,
        timeout=180,
        verify_ssl=False
    )
    session_id = sam.start_session(video_path)

    # Шаг 4: Отправляем промпты для размеченных кадров
    _log("🔎 Отправляем промпты только для размеченных кадров")
    reference_frames = []
    
    for idx, img_path in enumerate(image_paths):
        stem = img_path.stem
        yolo_lines = yolo_ann.get(stem, [])
        if not yolo_lines:
            continue

        prompts = yolo_to_prompts(yolo_lines, class_names)
        _log(f"Кадр {idx}: отправляем {len(prompts)} промптов")
        
        if len(prompts) > 0:
            try:
                resp = sam.send_prompt(frame_index=idx, prompts=prompts)
                reference_frames.append(idx)
                _log(f"✅ Промпты для кадра {idx} отправлены успешно")
                # Логируем ответ для отладки
                _log(f"   Ответ API: {json.dumps(resp, indent=2)[:500]}")
            except Exception as e:
                _log(f"❌ Ошибка при отправке промптов для кадра {idx}: {e}")

    # Шаг 5: Запускаем propagation
    if reference_frames:
        start_frame = reference_frames[0]
        _log(f"🚀 Запускаем propagation от кадра {start_frame}")
        try:
            prop_res = sam.propagate(
                start_frame_index=start_frame,
                direction="both",
                max_frames=len(image_paths)
            )
            _log(f"✅ Propagation завершён")
            _log(f"Результат propagation: {json.dumps(prop_res, indent=2)[:1000]}")
        except Exception as e:
            _log(f"❌ Ошибка propagation: {e}")
    else:
        _log("⚠️  Нет размеченных кадров для propagation")
        return

    # Шаг 6: Получаем ВСЕ результаты через API
    _log("📦 Получаем все результаты сессии")
    
    # Сначала смотрим, что есть в статусе
    status = sam.get_status()
    _log(f"DEBUG Status response: {json.dumps(status, indent=2)[:2000]}")
    
    # Пробуем получить результаты
    all_results = sam.get_all_results()
    _log(f"DEBUG All results: {json.dumps(all_results, indent=2)[:2000]}")
    
    # Извлекаем маски из результатов
    results_dir = OUTPUT_ROOT / "sam_results"
    results_dir.mkdir(exist_ok=True)
    
    # Парсим результаты в зависимости от их структуры
    all_masks = {}
    
    # Проверяем разные возможные структуры ответа
    if 'results' in all_results:
        results_data = all_results['results']
        _log(f"Найдены результаты в поле 'results'")
    elif 'frames' in all_results:
        results_data = all_results['frames']
        _log(f"Найдены результаты в поле 'frames'")
    elif 'objects' in all_results:
        results_data = all_results['objects']
        _log(f"Найдены результаты в поле 'objects'")
    else:
        # Может быть словарь с индексами кадров как ключами
        if isinstance(all_results, dict) and all_results:
            first_key = list(all_results.keys())[0]
            if first_key.isdigit() or isinstance(first_key, int):
                results_data = all_results
                _log(f"Результаты в формате словаря с индексами кадров")
            else:
                results_data = all_results
                _log(f"Неизвестная структура результатов, используем как есть")
        else:
            results_data = {}
            _log("⚠️ Не удалось найти результаты в ответе API")
    
    # Сохраняем полный ответ для анализа
    debug_file = results_dir / "api_response_debug.json"
    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    _log(f"💾 Полный ответ API сохранён в {debug_file}")
    
    # Пробуем извлечь маски для каждого кадра
    if isinstance(results_data, dict):
        for key, value in results_data.items():
            # Пытаемся определить индекс кадра
            if isinstance(key, str) and key.isdigit():
                frame_idx = int(key)
            elif isinstance(key, int):
                frame_idx = key
            elif 'frame_index' in value if isinstance(value, dict) else False:
                frame_idx = value['frame_index']
            else:
                continue
            
            if frame_idx < len(image_paths):
                stem = image_paths[frame_idx].stem
                mask_file = results_dir / f"{stem}_masks.json"
                with open(mask_file, "w", encoding="utf-8") as f:
                    json.dump(value, f)
                all_masks[frame_idx] = value
                
                # Проверяем наличие объектов
                objects = value.get('objects', []) if isinstance(value, dict) else []
                if objects:
                    _log(f"✅ Кадр {frame_idx}: найдено {len(objects)} объект(ов)")

    # Шаг 7: Визуализация
    _log("🖼️  Формируем финальное видео с визуализацией")
    vis_video_path = OUTPUT_ROOT / "visualized_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    first_frame = cv2.imread(str(image_paths[0]))
    frame_height, frame_width = first_frame.shape[:2]
    writer = cv2.VideoWriter(str(vis_video_path), fourcc, FPS, (frame_width, frame_height))

    frames_with_masks = 0
    for idx, img_path in enumerate(tqdm.tqdm(image_paths, desc="Визуализация")):
        frame = cv2.imread(str(img_path))
        stem = img_path.stem

        # Накладываем маски SAM
        frame_result = all_masks.get(idx, {})
        objects = frame_result.get('objects', []) if isinstance(frame_result, dict) else []
        
        if objects:
            frame = overlay_masks_on_frame(
                frame,
                objects,
                (frame.shape[0], frame.shape[1])
            )
            frames_with_masks += 1

        # Рисуем исходные YOLO boxes (опционально)
        yolo_lines = yolo_ann.get(stem, [])
        if yolo_lines:
            frame = draw_boxes_on_frame(frame, yolo_lines, class_names)

        writer.write(frame)

    writer.release()
    _log(f"✅ Финальное видео сохранено: {vis_video_path}")

    # Финальная статистика
    final_status = sam.get_status()
    _log(f"📊 Статистика сессии:")
    _log(f"   Статус: {final_status.get('status')}")
    _log(f"   Текущие объекты: {final_status.get('current_objects')}")
    _log(f"   Всего обработано кадров: {len(all_masks)}")
    _log(f"   Кадров с масками: {frames_with_masks}")
    
    if frames_with_masks == 0:
        _log("⚠️  ВНИМАНИЕ: Не найдено ни одного кадра с масками!")
        _log(f"   Проверьте файл {debug_file} чтобы понять структуру ответа API")


if __name__ == "__main__":
    run_pipeline()

class YOLOTrainer:
    def __init__(self, model_path,
                 data_path,
                 epochs=100, imgsz=640, batch=16):
        """
        Инициализация YOLOTrainer с параметрами обучения.

        Аргументы:
            model_path (str): Путь к весам модели YOLO
            data_path (str): Путь к файлу конфигурации данных YAML
            epochs (int): Количество эпох обучения
            imgsz (int): Размер изображения для обучения
            batch (int): Размер пакета (16 для GPU, например, T4)
        """
        self.model_path = model_path
        self.data_path = data_path
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.model = None
        self.train_results = None

    def load_model(self):
        """Загрузка модели YOLO."""
        self.model = YOLO(self.model_path)

    def train(self):
        """Обучение модели YOLOv11 с явными гиперпараметрами."""
        if self.model is None:
            self.load_model()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = self.model.train(
            data=self.data_path,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=device,
            lr0=0.001,
            optimizer="AdamW",
            weight_decay=0.0005,
        )
        return results

    def test(self, data_path=None, split="test", batch=16, conf=0.5):
        """Оценка модели на тестовом наборе данных."""
        if self.model is None:
            self.load_model()

        if data_path is None:
            data_path = self.data_path

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = self.model.val(
            data=data_path,
            imgsz=self.imgsz,
            batch=batch,
            split=split,
            conf=conf,
            device=device
        )

        return results

    def predict(self, source, conf=0.5, save=False):
        """Выполнение предсказания с использованием модели."""
        if self.model is None:
            self.load_model()
            
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        results = self.model.predict(
            source=source, 
            imgsz=self.imgsz, 
            conf=conf, save=save, 
            device=device
        )

        return results

    def save_model(self, save_path="/content/drive/MyDrive/tomato_leaf_11.pt"):
        """Сохранение обученной модели на Google Drive."""
        if self.model is None:
            return
        
        self.model.save(save_path)


import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

class YOLOv11Inference:
    def __init__(self, model_path,
                 device, class_labels, conf):
        """
        Инициализация модели YOLOv11 для выполнения предсказаний.

        Аргументы:
            model_path (str): Путь к обученной модели YOLOv11.
            device (str): Устройство для предсказаний ('cuda:0' или 'cpu').
            class_labels (list): Список меток классов для заболеваний.
            conf (float): Порог уверенности для предсказаний (0.0–1.0).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        if not class_labels:
            raise ValueError("Список меток классов не может быть пустым")

        self.device = device
        self.class_labels = class_labels
        self.conf = conf
        self.model = YOLO(model_path).to(self.device)

    def run(self, image_dir, n_ims=15, rows=3):
        """
        Выполняет предсказание на случайных изображениях из директории и отображает результаты в Colab.

        Аргументы:
            image_dir (str): Путь к директории с изображениями растений.
            n_ims (int): Количество изображений для визуализации.
            rows (int): Количество строк в сетке визуализации.

        Возвращает:
            list: Список словарей с результатами предсказаний (путь, рамки, метки, уверенности).
        """
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Директория не найдена: {image_dir}")

        image_extensions = (".jpg", ".jpeg", ".png")
        image_files = [f for f in Path(image_dir).glob("*") if f.suffix.lower() in image_extensions]

        if not image_files:
            raise ValueError(f"В директории {image_dir} нет изображений")

        n_ims = min(n_ims, len(image_files))
        selected_files = random.sample(image_files, n_ims)

        inference_results = self.model(selected_files, device=self.device, verbose=False, conf=self.conf)
        results_data = self.process_results(inference_results)
        self.visualize_results(inference_results, n_ims, rows)

        return results_data

    def predict_single(self, image_path):
        """
        Выполняет предсказание на одном изображении и возвращает результаты.

        Аргументы:
            image_path (str): Путь к изображению.

        Возвращает:
            dict: Словарь с результатами предсказания.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")

        results = self.model(image_path, device=self.device, verbose=False, conf=self.conf)

        return self.process_results(results)[0] if results else {}

    def process_results(self, results):
        """
        Обрабатывает результаты предсказания, извлекая рамки, метки и уверенности.

        Аргументы:
            results (list): Результаты предсказания от модели YOLO.

        Возвращает:
            list: Список словарей с данными о предсказаниях.
        """
        results_data = []
        for r in results:
            boxes = []
            for bbox in r.boxes:
                box = bbox.xyxy[0].cpu().numpy()
                class_id = int(bbox.cls[0])
                confidence = float(bbox.conf[0])
                boxes.append({
                    "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    "label": self.class_labels[class_id],
                    "confidence": confidence
                })
            results_data.append({"path": str(r.path), "boxes": boxes})
        return results_data

    def visualize_results(self, res, n_ims, rows):
      cols = (n_ims + rows - 1) // rows
      fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

      axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

      for idx, r in enumerate(res[:n_ims]):
          img = np.array(Image.open(r.path).convert("RGB"))
          axes[idx].imshow(img)

          for bbox in r.boxes:
              x1, y1, x2, y2 = map(int, bbox.xyxy[0])

              rect = plt.Rectangle(
                  (x1, y1), x2 - x1, y2 - y1,
                  fill=False, color='yellow', linewidth=2
              )
              axes[idx].add_patch(rect)

              label = f"{self.class_labels[int(bbox.cls[0])]} {float(bbox.conf[0]):.2f}"
              axes[idx].text(
                  x1, y1 - 10, label,
                  color='yellow', fontsize=12,
                  bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
              )

          axes[idx].axis('off')

      for idx in range(len(res), rows * cols):
          axes[idx].axis('off')

      plt.tight_layout()
      plt.show()
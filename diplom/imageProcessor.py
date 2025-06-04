from PIL import Image, ImageEnhance
import numpy as np
import cv2
import base64


class ImageProcessor:
    @staticmethod
    def _ensure_rgb_image(image):
        """
        Проверяет изображение и конвертирует в RGB, если необходимо.

        Args:
            image: Объект PIL.Image

        Returns:
            Изображение в формате RGB
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Входной параметр должен быть объектом PIL.Image")
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    @staticmethod
    def _resize_and_pad_image(image, target_size):
        """
        Изменяет размер изображения с сохранением пропорций и добавляет отступы.

        Args:
            image: Объект PIL.Image
            target_size: Целевой размер (ширина и высота)

        Returns:
            Изображение с измененным размером и отступами
        """
        width, height = image.size
        ratio = min(target_size / width, target_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        padded_image.paste(resized_image, ((target_size - new_width) // 2, (target_size - new_height) // 2))

        return padded_image

    @staticmethod
    def _enhance_image(image):
        """
        Применяет улучшения контрастности, резкости и цвета.

        Args:
            image: Объект PIL.Image

        Returns:
            Улучшенное изображение
        """
        enhanced_image = ImageEnhance.Contrast(image).enhance(1.0)
        enhanced_image = ImageEnhance.Sharpness(enhanced_image).enhance(0.5)
        enhanced_image = ImageEnhance.Color(enhanced_image).enhance(2.0)

        return enhanced_image

    @staticmethod
    def preprocess_image(image):
        """
        Обрабатывает изображение для предсказания: изменяет размер, добавляет отступы и улучшает качество.

        Args:
            image: Объект PIL.Image

        Returns:
            Обработанное изображение PIL.Image
        """
        image = ImageProcessor._ensure_rgb_image(image)
        resized_padded_image = ImageProcessor._resize_and_pad_image(image, target_size=640)
        enhanced_image = ImageProcessor._enhance_image(resized_padded_image)

        return enhanced_image

    @staticmethod
    def resize_for_display(image):
        """
        Изменяет размер изображения для отображения в веб-приложении.

        Args:
            image: Объект PIL.Image

        Returns:
            Измененное изображение PIL.Image
        """
        image = ImageProcessor._ensure_rgb_image(image)
        target_size = 500
        resized_image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)

        return resized_image

    @staticmethod
    def _convert_to_bgr_array(image):
        """
        Конвертирует изображение в numpy массив и преобразует в BGR.

        Args:
            image: Объект PIL.Image

        Returns:
            Массив BGR
        """
        image_array = np.array(image)
        bgr_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        return bgr_array

    @staticmethod
    def _encode_to_base64(bgr_array):
        """
        Кодирует массив BGR в строку base64.

        Args:
            bgr_array: Numpy массив в формате BGR

        Returns:
            Строка base64
        """
        _, buffer = cv2.imencode('.png', bgr_array)
        base64_string = base64.b64encode(buffer).decode('utf-8')

        return base64_string

    @staticmethod
    def convert_to_base64(image):
        """
        Конвертирует изображение в строку base64 для отображения в веб-приложении.

        Args:
            image: Объект PIL.Image

        Returns:
            Строка base64
        """
        bgr_array = ImageProcessor._convert_to_bgr_array(image)
        base64_string = ImageProcessor._encode_to_base64(bgr_array)
        
        return base64_string
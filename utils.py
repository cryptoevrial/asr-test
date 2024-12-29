import ffmpeg
from abc import ABC, abstractmethod
import gigaam
from dotenv import load_dotenv
import os
import torch
import whisper
from typing import Optional, Any
from pathlib import Path
import json
import gc


class BaseModelASR(ABC):
    """
    Абстрактный базовый класс для ASR моделей.
    Определяет общий интерфейс для всех реализаций ASR.

    :attr device (str): Устройство для вычислений ('cuda:N' или 'cpu')
    :attr model: Экземпляр модели
    :attr model_name (str): Название загруженной модели
    """

    def __init__(self, model_name: str, gpu_id: int):
        """
        Инициализация базового класса ASR.

        :param model_name название модели для загрузки
        :param gpu_id номер gpu на который будет загружена модель
        """
        if gpu_id >= torch.cuda.device_count() and torch.cuda.is_available():
            raise ValueError(f"GPU с индексом {gpu_id} не найден. "
                           f"Доступно устройств: {torch.cuda.device_count()}")
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model: Optional[Any] = None
        self.model_name: str = model_name
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Проверяет, загружена ли модель."""
        return self._is_loaded

    @abstractmethod
    def load_model(self, model_name: str):
        """
        Загрузка модели на устройство
        :param model_name: название модели
        :return: экземпляр модели
        """
        pass

    @abstractmethod
    def transcribe(self, audio_path: str):
        """
        Переводит аудио в текст и возвращает список словарей
        с текстом и временными метками
        :param audio_path: путь к аудио файлу
        :return: список словарей с текстом и временными метками
        """
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена. Сначала вызовите load_model()")

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")

    @abstractmethod
    def get_json(self, transcription_result: list):
        """
        Сохраняет результат транскрибации в файл по указанному пути
        :param transcription_result
        :return: json
        """
        pass

    def cleanup(self):
        """
        Очищает память от загруженной модели и вычислений
        :return:
        """
        try:
            if self.model:
                del self.model
                self.model = None
                self._is_loaded = False

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print("Модель успешно выгружена из памяти")

        except Exception as e:
            print(f"Ошибка при очистке памяти: {e}")
            raise


class Whisper(BaseModelASR):
    """
    Реализация ASR на базе Whisper модели.
    """
    def __init__(self, model_name: str, gpu_id: int):
        super().__init__(model_name, gpu_id)
        self.available_models = {
            "tiny", "base", "small", "medium", "large"
        }

    def load_model(self, model_name: str):
        """
        Загружает модель Whisper.

        :param model_name: название модели whisper (tiny, base, small, medium, large)
        """
        try:
            if model_name not in self.available_models:
                raise ValueError(
                    f"Недопустимое название модели. Доступные модели: {self.available_models}"
                )
            self.model = whisper.load_model(model_name).to(self.device)
            self._is_loaded = True
            print(f"Модель {model_name} успешно загружена на {self.device}")
            return self.model

        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            raise

    def transcribe(self, audio_path: str) -> Optional[dict]:
        super().transcribe(audio_path)

        try:
            transcribe_options = {
                "task": "transcribe",
                "fp16": True,
                "verbose": None,
                "language": "ru"
            }
            print(f"Start transcribing {audio_path}")
            result = self.model.transcribe(str(audio_path), **transcribe_options)
            print("Transcribing completed")
            return result

        except Exception as e:
            print(f"Error during transcription: {e}")
            raise

    def get_json(self, transcription_result: dict):
        if not transcription_result or "segments" not in transcription_result:
            raise ValueError("Invalid transcription result format")

        json_obj = {}
        for index, segment in enumerate(transcription_result["segments"]):
            json_obj[index] = {
                "start": float(f'{segment["start"]:.2f}'),
                "end": float(f'{segment["end"]:.2f}'),
                "text": segment["text"].strip()
            }
        return json.dumps(json_obj, ensure_ascii=False, indent=2)


class GigaAM(BaseModelASR):
    """
    Реализация ASR на базе модели GigaAM
    """

    def __init__(self, model_name: str, gpu_id: int):
        super().__init__(model_name, gpu_id)
        self.available_models = {"ctc", "rnnt"}

    def load_model(self, model_name: str):
        """
        Загружает модель Whisper.

        :param model_name: название модели GigaAM ("ctc", "rnnt")
        """
        try:
            if model_name not in self.available_models:
                raise ValueError(
                    f"Недопустимое название модели. Доступные модели: {self.available_models}"
                )
            self.model = gigaam.load_model(model_name=model_name,
                                           device=self.device
                                           )
            self._is_loaded = True
            print(f"Модель {model_name} успешно загружена на {self.device}")
            return self.model

        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            raise

    def transcribe(self, audio_path: str):
        try:
            super().transcribe(audio_path)

            if self.load_hf_token():
                print(f"Start transcribing {audio_path}")
                result = self.model.transcribe_longform(audio_path)
                print("Transcribing completed")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                return result
        except Exception as e:
            print(f"Ошибка при транскрибации: {e}")
            raise
        finally:
            # Принудительная очистка после завершения
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def get_json(self, transcription_result: list):
        if not transcription_result:
            raise ValueError("Invalid transcription result format")

        json_obj = {}
        for index, segment in enumerate(transcription_result):
            json_obj[index] = {
                "start": float(f'{segment["boundaries"][0]:.2f}'),
                "end": float(f'{segment["boundaries"][1]:.2f}'),
                "text": segment["transcription"].strip(),
            }
        return json.dumps(json_obj, ensure_ascii=False, indent=2)

    @staticmethod
    def load_hf_token():
        load_dotenv()  # Всегда загружаем переменные окружения
        token = os.getenv("HF_TOKEN")
        if not token:
            print("Set env 'HF_TOKEN' for transcribing")
        return token

    def cleanup(self):
        """
        Очищает память от загруженной модели и вычислений с учетом архитектуры GigaAM
        """
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Очищаем компоненты модели
                if hasattr(self.model, 'preprocessor'):
                    del self.model.preprocessor
                if hasattr(self.model, 'encoder'):
                    del self.model.encoder

                # Специфичные компоненты для GigaAMASR
                if hasattr(self.model, 'head'):
                    if hasattr(self.model.head, 'decoder'):
                        del self.model.head.decoder
                    if hasattr(self.model.head, 'joint'):
                        del self.model.head.joint
                    del self.model.head

                if hasattr(self.model, 'decoding'):
                    del self.model.decoding

                # Очищаем CUDA кэш перед основным удалением
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Переносим на CPU и удаляем
                self.model.cpu()
                del self.model
                self.model = None
                self._is_loaded = False

                # Финальная очистка
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                print("Модель GigaAM успешно выгружена из памяти")
        except Exception as e:
            print(f"Ошибка при очистке памяти: {e}")
            raise


def print_torch_cuda_info():
    print("CUDA доступна:", torch.cuda.is_available())
    print("Количество GPU:", torch.cuda.device_count())
    print("Текущее GPU устройство:", torch.cuda.current_device())
    print("Название GPU:", torch.cuda.get_device_name(0))
    print("Версия torch:", torch.__version__)
    print("Версия cuda:",torch.version.cuda)
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.2f}GB")


def clear_gpu_memory():
    """Очистка памяти GPU"""
    if torch.cuda.is_available():
        # Очищаем кэш PyTorch CUDA
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        # Запускаем сборщик мусора Python
        gc.collect()
        print("GPU память очищена")


# переписать функцию для принятия json
def split_audio(input_audio, output_dir, segments):
    """Split audio file into segments based on timestamps."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, segment in enumerate(segments, 1):
        audio_path = Path(input_audio)
        file_name = audio_path.stem
        output_file = os.path.join(output_dir, f'{file_name}_{i:03d}.mp3')

        try:
            stream = ffmpeg.input(input_audio, ss=float(segment['start']), t=float(segment['end'])-float(segment['start']))
            stream = ffmpeg.output(stream, output_file)
            # Debug
            #cmd = ffmpeg.compile(stream)
            #print(f"FFmpeg command: {' '.join(cmd)}")
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

            print(f"Created segment {i}: {segment['text'][:50]}...")
        except ffmpeg.Error as e:
            print(f"Error processing segment {i}: {str(e)}")


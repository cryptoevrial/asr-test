from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class AudioFiles(Base):
    """
    Таблица аудифайлов
    """
    __tablename__ = "AudioFiles"

    id = Column(Integer, primary_key=True)

    name = Column(String, nullable=False, comment="Имя файла")
    path_to_file = Column(String, nullable=False, comment="Путь к файлу")
    duration = Column(Float, nullable=False, comment="Длительность аудиофайла в секундах")
    size = Column(Integer, nullable=False, comment="Размер файла в байтах")

    segments = relationship("AudioSegments", back_populates="file")


class AudioSegments(Base):
    """
    Таблица с распознанным текстом по сегментам
    """
    __tablename__ = "AudioSegments"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, comment="Имя файла")
    path_to_file = Column(String, nullable=False, comment="Путь к файлу")
    start_time = Column(Float, comment="Время начала сегмента")
    end_time = Column(Float, comment="Время конца сегмента")
    text = Column(String, comment="Транскрипция сегмента")
    from_model = Column(String, comment="Модель транскрибации")

    file_id = Column(Integer, ForeignKey("AudioFiles.id"), nullable=False)

    file = relationship("AudioFiles", back_populates="segments")
    edited_text = relationship("Labeling", back_populates="segment")


class Labeling(Base):
    """
    Таблица с разметкой
    """
    __tablename__ = "Labeling"

    id = Column(Integer, primary_key=True)
    edited_text = Column(String, comment="Отредактированный текст")
    editor = Column(String, comment="Пользователь, который редактировал текст")
    edited_time = Column(DateTime, comment="Время редактирования")
    is_edited = Column(Boolean)
    abbreviation = Column(Boolean, comment="Есть ли аббревиатуры")
    not_sure = Column(Boolean, comment="Уверенность в тексте")
    comment = Column(String, comment="Дополнительные комментарии")

    segment_id = Column(Integer, ForeignKey("AudioSegments.id"), nullable=False)

    segment = relationship("AudioSegments", back_populates="edited_text")

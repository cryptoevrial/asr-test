import json
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import AudioFiles, Base

DATABASE_URL = "sqlite:///sqlite_db.db"


class DatabaseSession:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Генератор сессий для использования в контекстном менеджере"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

def add_all_objs(db: DatabaseSession, objs: list):
    with db.get_session() as session:
        session.add_all(objs)

def add_segment_from_results(db: DatabaseSession, results: list):
    for result in results:
        with db.get_session() as session:
            file_in_db = session.query(AudioFiles).filter(AudioFiles.name == result["file_name"]).first()
            file_id = file_in_db.id
            json_dict = json.loads(result["json"])


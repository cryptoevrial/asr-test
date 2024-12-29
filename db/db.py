from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from typing import Generator
from models import Base

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

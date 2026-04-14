import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

Base = declarative_base()

class DbConnection:
    """
    So from this "Singleton"(design pattern)
    database connection manager,
    No matter how many times DbConnection() is called,
    only ONE instance will ever exist. This guarantees
    one engine, one session factory, one connection pool.

    Checks first for DATABASE_URL from .env.
    If PostgreSQL is present, use it. Otherwise, fallback to SQLite.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.database_url = os.getenv("DATABASE_URL")

        if self.database_url:
            # PostgreSQL I had set from my .env file
            self.engine = create_engine(self.database_url)
        else:
            # Fallback but to SQLite, stored but as a local file
            self.database_url = "sqlite:///./church_pipeline.db"
            self.engine = create_engine(self.database_url, connect_args={"check_same_thread": False})

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self._initialized = True

    def get_db(self):
        # The FastAPI dependency yields a session, and closes it when it's done
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()


# Singleton instance — even if someone calls DbConnection() elsewhere,
# they get this same object back.
db_connection = DbConnection()
engine = db_connection.engine
SessionLocal = db_connection.SessionLocal
get_db = db_connection.get_db

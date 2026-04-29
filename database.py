import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

Base = declarative_base()

class DbConnection:
    """
    My DB connection manager, written as a Singleton on purpose so that no
    matter where DbConnection() gets called from in the project, I'm always
    talking to the same engine, the same session factory, and the same
    connection pool.

    I look for DATABASE_URL in .env first; if it's there I use Postgres,
    otherwise I drop down to a local SQLite file so the app still runs even
    when nothing is configured.
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
            # If I set DATABASE_URL in .env, use that (Postgres in my case).
            self.engine = create_engine(self.database_url)
        else:
            # Otherwise just use a local SQLite file — keeps things simple.
            self.database_url = "sqlite:///./church_pipeline.db"
            self.engine = create_engine(self.database_url, connect_args={"check_same_thread": False})

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self._initialized = True

    def get_db(self):
        # FastAPI dependency: hand out a session, then close it once the
        # request is done so I don't leak connections.
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()


# Module-level handles. Anyone calling DbConnection() from elsewhere gets
# the same instance back, so these stay valid project-wide.
db_connection = DbConnection()
engine = db_connection.engine
SessionLocal = db_connection.SessionLocal
get_db = db_connection.get_db

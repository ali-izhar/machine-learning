import os
import secrets
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
time_to_live = 24


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", default=secrets.token_hex(16))
    SESSION_TYPE = "filesystem"

    # Uploads settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", "uploads")
    ALLOWED_EXTENSIONS = {"csv"}
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Model settings
    MODEL_PATH = os.path.join("api", "models")

    # Creating flask_session directory if it doesn't exist
    SESSION_FILE_DIR = os.path.join(BASE_DIR, "..", "flask_session")
    os.makedirs(SESSION_FILE_DIR, exist_ok=True)

    PERMANENT_SESSION_LIFETIME = timedelta(hours=time_to_live)

    # Database settings
    DB_NAME = "db.sqlite3"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Creating instance directory if it doesn't exist
    INSTANCE_DIR = os.path.join(BASE_DIR, "..", "instance")
    os.makedirs(INSTANCE_DIR, exist_ok=True)

    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", default=f"sqlite:///{os.path.join(INSTANCE_DIR, DB_NAME)}"
    )

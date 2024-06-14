from app.models import User
from app import db
from werkzeug.security import generate_password_hash, check_password_hash
import logging


def create_user(username, email, password):
    try:
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()
        return user
    except Exception as e:
        logging.error(f"Error creating user: {e}")
        return None


def get_user_by_email(email):
    try:
        return User.query.filter_by(email=email).first()
    except Exception as e:
        logging.error(f"Error getting user by email: {e}")
        return None


def get_user_by_id(user_id):
    try:
        return User.query.get(user_id)
    except Exception as e:
        logging.error(f"Error getting user by id: {e}")
        return None


def check_password(user, password):
    return check_password_hash(user.password_hash, password)


def update_user(user, **kwargs):
    try:
        for key, value in kwargs.items():
            setattr(user, key, value)
        db.session.commit()
        return user
    except Exception as e:
        logging.error(f"Error updating user: {e}")
        return None


def delete_user(user):
    try:
        db.session.delete(user)
        db.session.commit()
        return True
    except Exception as e:
        logging.error(f"Error deleting user: {e}")
        return False

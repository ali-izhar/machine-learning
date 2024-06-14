from flask import Flask, render_template, redirect, url_for, flash
from flask_login import LoginManager
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from config.settings import Config

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth_bp.login"
login_manager.login_message = "Please login to access this page."
login_manager.login_message_category = "info"


def create_app(config_class=Config):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    Session(app)

    # Import and register blueprints
    from .views.auth import auth_bp
    from .views.main import main_bp
    from .views.user import user_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(user_bp)

    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template("500.html"), 500

    # Handle unauthorized access
    @login_manager.unauthorized_handler
    def unauthorized():
        flash("You must be logged in to view that page.", "info")
        return redirect(url_for("auth_bp.login"))

    # Create tables before the first request
    @app.before_request
    def create_tables():
        with app.app_context():
            db.create_all()

    # User loader callback
    @login_manager.user_loader
    def load_user(user_id):
        from app.models import User

        return User.query.get(int(user_id))

    return app

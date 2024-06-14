from flask import request, render_template, redirect, url_for, flash, Blueprint
from flask_login import current_user, login_user, logout_user, login_required
from app.services.db_ops import create_user, get_user_by_email, check_password

auth_bp = Blueprint("auth_bp", __name__, url_prefix="/auth")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main_bp.index"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = get_user_by_email(email)

        if user and check_password(user, password):
            login_user(user, remember=True)
            flash("Successfully logged in!", "success")
            return redirect(url_for("main_bp.index"))
        else:
            flash("Invalid email or password!", "warning")

    return render_template("signin.html", user=current_user)


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("main_bp.index"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        username = email.split("@")[0]

        if get_user_by_email(email):
            flash("User already exists!", "warning")
        else:
            user = create_user(username, email, password)
            if user:
                login_user(user, remember=True)
                flash("Successfully signed up!", "success")
                return redirect(url_for("main_bp.index"))
            else:
                flash("Error signing up!", "danger")

    return render_template("signup.html", user=current_user)


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Successfully logged out!", "info")
    return redirect(url_for("main_bp.index"))

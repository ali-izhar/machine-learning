from flask import Blueprint, request, render_template, flash, redirect, url_for
from flask_login import current_user, login_required
from app.services.db_ops import update_user, delete_user, check_password
from werkzeug.security import generate_password_hash

user_bp = Blueprint("user_bp", __name__, url_prefix="/user")


@user_bp.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user=current_user)


@user_bp.route("/edit_profile", methods=["GET", "POST"])
@login_required
def edit_profile():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        bio = request.form.get("bio")

        updated_user = update_user(
            current_user, username=username, email=email, bio=bio
        )

        if updated_user:
            flash("Profile updated successfully!", "success")
            return redirect(url_for("user_bp.profile"))
        else:
            flash("Error updating profile.", "danger")

    return render_template("edit_profile.html", user=current_user)


@user_bp.route("/change_password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        current_password = request.form.get("current_password")
        new_password = request.form.get("new_password")
        confirm_new_password = request.form.get("confirm_new_password")

        if not check_password(current_user, current_password):
            flash("Current password is incorrect.", "danger")
        elif new_password != confirm_new_password:
            flash("New passwords do not match.", "danger")
        else:
            current_user.password_hash = generate_password_hash(new_password)
            update_user(current_user, password_hash=current_user.password_hash)
            flash("Password changed successfully!", "success")
            return redirect(url_for("user_bp.profile"))

    return render_template("change_password.html", user=current_user)


@user_bp.route("/delete_account", methods=["POST"])
@login_required
def delete_account():
    if request.method == "POST":
        if delete_user(current_user):
            flash("Account deleted successfully.", "info")
            return redirect(url_for("auth_bp.logout"))
        else:
            flash("Error deleting account.", "danger")

    return redirect(url_for("user_bp.profile"))

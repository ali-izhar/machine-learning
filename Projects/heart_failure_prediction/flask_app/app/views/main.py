from flask import Blueprint, render_template, session, request, jsonify, flash
from flask_login import current_user, login_required
import pandas as pd
import joblib
from io import StringIO
from config.settings import Config

main_bp = Blueprint("main_bp", __name__, url_prefix="/")

MODEL_PATH = Config.MODEL_PATH
# MODEL_NAME = "random_forest.pkl"
MODEL_NAME = "xgboost.pkl"
model = joblib.load(f"{MODEL_PATH}/{MODEL_NAME}")


# Define the feature columns expected by the model
EXPECTED_FEATURES = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    cat_variables = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    for var in cat_variables:
        if var not in data.columns:
            raise ValueError(f"Missing column in the uploaded data: {var}")

    # Convert categorical columns to type 'category' before encoding
    data[cat_variables] = data[cat_variables].apply(lambda x: x.astype("category"))

    # Get dummies for categorical columns
    data = pd.get_dummies(data, prefix=cat_variables, columns=cat_variables)

    # Ensure that all expected features are present after preprocessing
    expected_dummies = [
        "Sex_F",
        "Sex_M",
        "ChestPainType_ASY",
        "ChestPainType_ATA",
        "ChestPainType_NAP",
        "ChestPainType_TA",
        "RestingECG_LVH",
        "RestingECG_Normal",
        "RestingECG_ST",
        "ExerciseAngina_N",
        "ExerciseAngina_Y",
        "ST_Slope_Down",
        "ST_Slope_Flat",
        "ST_Slope_Up",
    ]

    for col in expected_dummies:
        if col not in data.columns:
            data[col] = 0

    # Drop any columns that are not part of the model features
    model_features = EXPECTED_FEATURES + expected_dummies
    data = data[[col for col in data.columns if col in model_features]]

    return data


@main_bp.route("/")
@login_required
def index():
    session.permanent = False
    return render_template("index.html", user=current_user)


@main_bp.route("/dashboard")
@login_required
def dashboard():
    username = current_user.email
    username = username.split("@")[0].title()
    return render_template("dashboard.html", user=current_user, username=username)


@main_bp.route("/upload_data", methods=["POST"])
@login_required
def upload_data():
    if "data_file" not in request.files:
        return jsonify({"message": "No file part", "category": "danger"}), 400
    file = request.files["data_file"]
    if file.filename == "":
        return jsonify({"message": "No selected file", "category": "danger"}), 400
    if file and allowed_file(file.filename):
        try:
            data = pd.read_csv(file)
            data.columns = data.columns.str.strip()
            data = preprocess_data(data)
            session["uploaded_data"] = data.to_json(orient="split")
            return (
                jsonify(
                    {"message": "Data uploaded successfully!", "category": "success"}
                ),
                200,
            )
        except pd.errors.ParserError:
            return (
                jsonify(
                    {
                        "message": "Error parsing CSV file. Please ensure it is correctly formatted.",
                        "category": "danger",
                    }
                ),
                400,
            )
        except Exception as e:
            print("Error processing data:", e)
            return (
                jsonify(
                    {"message": f"Error processing data: {e}", "category": "danger"}
                ),
                400,
            )
    return jsonify({"message": "Invalid file type", "category": "danger"}), 400


@main_bp.route("/manual_input_data", methods=["POST"])
@login_required
def manual_input_data():
    try:
        data = {
            "Age": int(request.form.get("age")),
            "Sex": request.form.get("sex"),
            "ChestPainType": request.form.get("chestPainType"),
            "RestingBP": int(request.form.get("restingBP")),
            "Cholesterol": int(request.form.get("cholesterol")),
            "FastingBS": (
                int(request.form.get("fastingBS"))
                if request.form.get("fastingBS")
                else 0
            ),
            "RestingECG": request.form.get("restingECG"),
            "MaxHR": int(request.form.get("maxHR")),
            "ExerciseAngina": request.form.get("exerciseAngina"),
            "Oldpeak": float(request.form.get("oldpeak")),
            "ST_Slope": request.form.get("st_slope"),
        }
        df = pd.DataFrame([data])
        df.columns = df.columns.str.strip()
        df = preprocess_data(df)
        session["uploaded_data"] = df.to_json(orient="split")
        return (
            jsonify({"message": "Data inputted successfully!", "category": "success"}),
            200,
        )
    except Exception as e:
        return (
            jsonify({"message": f"Error processing data: {e}", "category": "danger"}),
            400,
        )


@main_bp.route("/analyze_data", methods=["POST"])
@login_required
def analyze_data():
    try:
        if "uploaded_data" in session:
            data_json = session["uploaded_data"]
            data = pd.read_json(StringIO(data_json), orient="split")
            predictions = model.predict(data)
            data["predictions"] = predictions

            # Convert the data to JSON format
            result_json = data.to_json(orient="records")
            return jsonify({"status": "success", "result": result_json}), 200
        else:
            flash("No data uploaded for analysis", "danger")
            return (
                jsonify(
                    {"status": "error", "message": "No data uploaded for analysis"}
                ),
                400,
            )
    except Exception as e:
        flash(f"Error analyzing data: {str(e)}", "danger")
        return (
            jsonify({"status": "error", "message": f"Error analyzing data: {str(e)}"}),
            400,
        )


@main_bp.route("/view_stats")
@login_required
def view_stats():
    stats = [
        {"date": "2023-06-01", "result": "Sample result 1"},
        {"date": "2023-06-02", "result": "Sample result 2"},
    ]
    return render_template("stats.html", stats=stats, user=current_user)

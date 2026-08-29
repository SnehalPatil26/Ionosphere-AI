import os
import traceback
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template, request

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "prediction_history.db"
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

dataset = None

extra_trees_model = None
knn_model = None
scaler = None

model_mae = None

dataset_min_date = None
dataset_max_date = None

application_initialized = False


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            location TEXT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            tec REAL NOT NULL,
            activity TEXT,
            risk TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()

    print("Prediction history database: READY")


# ============================================================
# SAVE PREDICTION HISTORY
# ============================================================

def save_prediction_history(
    latitude,
    longitude,
    location,
    date_string,
    time_string,
    predicted_tec,
    activity,
    risk
):

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            latitude,
            longitude,
            location,
            date,
            time,
            tec,
            activity,
            risk,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            float(latitude),
            float(longitude),
            str(location),
            str(date_string),
            str(time_string),
            float(predicted_tec),
            str(activity),
            str(risk),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# GET PREDICTION HISTORY
# ============================================================

def get_prediction_history(limit=20):

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            latitude,
            longitude,
            location,
            date,
            time,
            tec,
            activity,
            risk,
            created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),)
    )

    rows = cursor.fetchall()

    connection.close()

    predictions = []

    for row in rows:

        predictions.append(
            {
                "id": row["id"],
                "latitude": round(float(row["latitude"]), 4),
                "longitude": round(float(row["longitude"]), 4),
                "location": row["location"] or "Unknown",
                "date": row["date"],
                "time": row["time"],
                "tec": round(float(row["tec"]), 4),
                "activity": row["activity"] or "-",
                "risk": row["risk"] or "-",
                "created_at": row["created_at"]
            }
        )

    return predictions


# ============================================================
# GET PREDICTION COUNT
# ============================================================

def get_prediction_count():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        """
    )

    result = cursor.fetchone()

    connection.close()

    if result is None:
        return 0

    return int(result[0])


# ============================================================
# SELECT DATASET
# ============================================================

def get_dataset_path():

    preferred_files = [

        "tec_omni_ml_dataset.csv",
        "tec_all_years.csv"

    ]

    for filename in preferred_files:

        file_path = os.path.join(
            DATA_FOLDER,
            filename
        )

        if os.path.exists(file_path):

            return file_path

    raise FileNotFoundError(
        "Dataset not found.\n"
        "Expected one of:\n"
        "tec_omni_ml_dataset.csv\n"
        "tec_all_years.csv"
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    global dataset
    global dataset_min_date
    global dataset_max_date

    print("\n" + "=" * 70)
    print("IONOSPHERE-AI DATASET")
    print("=" * 70)

    dataset_path = get_dataset_path()

    print(
        f"Loading ML dataset: "
        f"{os.path.basename(dataset_path)}"
    )

    dataset = pd.read_csv(
        dataset_path,
        low_memory=False
    )

    print(f"Records found: {len(dataset)}")

    print("\nColumns:")
    print(dataset.columns.tolist())

    # --------------------------------------------------------
    # STANDARDIZE COLUMN NAMES
    # --------------------------------------------------------

    dataset.columns = [
        str(column).strip().lower()
        for column in dataset.columns
    ]

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "latitude",
        "longitude",
        "tec"
    ]

    for column in required_columns:

        if column not in dataset.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    # --------------------------------------------------------
    # CREATE DATE COLUMN
    # --------------------------------------------------------

    if "date" in dataset.columns:

        dataset["date"] = pd.to_datetime(
            dataset["date"],
            errors="coerce"
        )

    elif "datetime" in dataset.columns:

        dataset["date"] = pd.to_datetime(
            dataset["datetime"],
            errors="coerce"
        )

    elif "timestamp" in dataset.columns:

        dataset["date"] = pd.to_datetime(
            dataset["timestamp"],
            errors="coerce"
        )

    elif all(
        column in dataset.columns
        for column in ["year", "month", "day"]
    ):

        dataset["date"] = pd.to_datetime(
            {
                "year": pd.to_numeric(
                    dataset["year"],
                    errors="coerce"
                ),

                "month": pd.to_numeric(
                    dataset["month"],
                    errors="coerce"
                ),

                "day": pd.to_numeric(
                    dataset["day"],
                    errors="coerce"
                )
            },
            errors="coerce"
        )

    else:

        print(
            "WARNING: No date column found. "
            "Using fallback date."
        )

        dataset["date"] = pd.Timestamp(
            "2020-01-01"
        )

    # --------------------------------------------------------
    # CREATE TIME FEATURES
    # --------------------------------------------------------

    if "time" in dataset.columns:

        time_values = pd.to_datetime(
            dataset["time"].astype(str),
            errors="coerce"
        )

        dataset["hour"] = (
            time_values.dt.hour.fillna(12)
        )

        dataset["minute"] = (
            time_values.dt.minute.fillna(0)
        )

    else:

        if "hour" in dataset.columns:

            dataset["hour"] = pd.to_numeric(
                dataset["hour"],
                errors="coerce"
            ).fillna(12)

        else:

            dataset["hour"] = 12

        if "minute" in dataset.columns:

            dataset["minute"] = pd.to_numeric(
                dataset["minute"],
                errors="coerce"
            ).fillna(0)

        else:

            dataset["minute"] = 0

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    dataset["latitude"] = pd.to_numeric(
        dataset["latitude"],
        errors="coerce"
    )

    dataset["longitude"] = pd.to_numeric(
        dataset["longitude"],
        errors="coerce"
    )

    dataset["tec"] = pd.to_numeric(
        dataset["tec"],
        errors="coerce"
    )

    dataset["hour"] = pd.to_numeric(
        dataset["hour"],
        errors="coerce"
    ).fillna(12)

    dataset["minute"] = pd.to_numeric(
        dataset["minute"],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # REMOVE INVALID VALUES
    # --------------------------------------------------------

    dataset.replace(
        [
            np.inf,
            -np.inf,
            1e31,
            -1e31,
            1e30,
            -1e30
        ],
        np.nan,
        inplace=True
    )

    dataset.loc[
        dataset["tec"] <= 0,
        "tec"
    ] = np.nan

    dataset.loc[
        dataset["tec"] > 100,
        "tec"
    ] = np.nan

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    dataset["date"] = pd.to_datetime(
        dataset["date"],
        errors="coerce"
    )

    dataset["year"] = (
        dataset["date"].dt.year
    )

    dataset["month"] = (
        dataset["date"].dt.month
    )

    dataset["day"] = (
        dataset["date"].dt.day
    )

    dataset["dayofyear"] = (
        dataset["date"].dt.dayofyear
    )

    # --------------------------------------------------------
    # TIME DECIMAL
    # --------------------------------------------------------

    dataset["time_decimal"] = (
        dataset["hour"]
        +
        dataset["minute"] / 60
    )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    dataset.dropna(
        subset=[
            "latitude",
            "longitude",
            "tec",
            "date",
            "year",
            "month",
            "day",
            "dayofyear"
        ],
        inplace=True
    )

    dataset.reset_index(
        drop=True,
        inplace=True
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "Dataset contains no valid records."
        )

    # --------------------------------------------------------
    # DATE RANGE
    # --------------------------------------------------------

    dataset_min_date = dataset["date"].min()

    dataset_max_date = dataset["date"].max()

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 70)

    print(f"Records: {len(dataset)}")

    print(
        f"TEC Min: "
        f"{dataset['tec'].min():.4f}"
    )

    print(
        f"TEC Max: "
        f"{dataset['tec'].max():.4f}"
    )

    print(
        f"TEC Average: "
        f"{dataset['tec'].mean():.4f}"
    )

    print(
        f"Date Range: "
        f"{dataset_min_date.date()} "
        f"to "
        f"{dataset_max_date.date()}"
    )

    print("=" * 70)


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    global extra_trees_model
    global knn_model
    global scaler
    global model_mae

    if dataset is None:

        raise RuntimeError(
            "Dataset is not loaded."
        )

    print("\n" + "=" * 70)
    print("TRAINING IONOSPHERE-AI MODEL")
    print("=" * 70)

    full_features = [

        "latitude",
        "longitude",

        "year",
        "month",
        "day",

        "hour",
        "minute",

        "dayofyear",

        "time_decimal"

    ]

    knn_features = [

        "latitude",
        "longitude",

        "dayofyear",

        "time_decimal"

    ]

    X = dataset[
        full_features
    ].copy()

    y = dataset[
        "tec"
    ].copy()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    # --------------------------------------------------------
    # EXTRA TREES
    # --------------------------------------------------------

    print("Training ExtraTrees...")

    extra_trees_model = ExtraTreesRegressor(

        n_estimators=300,

        max_depth=25,

        min_samples_leaf=1,

        random_state=42,

        n_jobs=-1

    )

    extra_trees_model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    print("Training Distance KNN...")

    scaler = StandardScaler()

    X_knn = dataset[
        knn_features
    ].copy()

    X_knn_scaled = scaler.fit_transform(
        X_knn
    )

    knn_model = KNeighborsRegressor(

        n_neighbors=8,

        weights="distance",

        n_jobs=-1

    )

    knn_model.fit(
        X_knn_scaled,
        y
    )

    # --------------------------------------------------------
    # MODEL MAE
    # --------------------------------------------------------

    test_prediction = (
        extra_trees_model.predict(
            X_test
        )
    )

    model_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    print(
        f"Model MAE: "
        f"{model_mae:.4f}"
    )

    print(
        "Hybrid model trained successfully."
    )

    print("=" * 70)


# ============================================================
# IONOSPHERE CLASSIFICATION
# ============================================================

def classify_ionosphere(tec_value):

    tec_value = float(tec_value)

    if tec_value < 5:

        return (
            "LOW TEC",
            "LOW"
        )

    elif tec_value < 10:

        return (
            "MEDIUM TEC",
            "MEDIUM"
        )

    else:

        return (
            "HIGH TEC",
            "HIGH"
        )


# ============================================================
# CREATE PREDICTION FEATURES
# ============================================================

def create_prediction_features(
    latitude,
    longitude,
    date_string,
    time_string
):

    datetime_object = datetime.strptime(

        f"{date_string} {time_string}",

        "%Y-%m-%d %H:%M"

    )

    year = datetime_object.year
    month = datetime_object.month
    day = datetime_object.day

    hour = datetime_object.hour
    minute = datetime_object.minute

    dayofyear = (
        datetime_object
        .timetuple()
        .tm_yday
    )

    time_decimal = (
        hour
        +
        minute / 60
    )

    full_input = pd.DataFrame(

        [[

            latitude,
            longitude,

            year,
            month,
            day,

            hour,
            minute,

            dayofyear,

            time_decimal

        ]],

        columns=[

            "latitude",
            "longitude",

            "year",
            "month",
            "day",

            "hour",
            "minute",

            "dayofyear",

            "time_decimal"

        ]

    )

    knn_input = pd.DataFrame(

        [[

            latitude,
            longitude,

            dayofyear,

            time_decimal

        ]],

        columns=[

            "latitude",
            "longitude",

            "dayofyear",

            "time_decimal"

        ]

    )

    return (
        full_input,
        knn_input
    )


# ============================================================
# APPLICATION STARTUP
# IMPORTANT FOR LOCAL + RENDER DEPLOYMENT
# ============================================================

def initialize_application():

    global application_initialized

    if application_initialized:
        return

    print("\n" + "=" * 70)
    print("INITIALIZING IONOSPHERE-AI APPLICATION")
    print("=" * 70)

    initialize_database()

    load_dataset()

    train_model()

    application_initialized = True

    print("=" * 70)
    print("APPLICATION INITIALIZATION COMPLETE")
    print("=" * 70 + "\n")


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route("/prediction")
def prediction():

    return render_template(
        "prediction.html"
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================

@app.route("/dashboard")
def dashboard():

    global dataset

    try:

        initialize_application()

        if dataset is None:

            raise RuntimeError(
                "Dataset is not loaded."
            )

        if dataset.empty:

            raise RuntimeError(
                "Dataset is empty."
            )

        # ----------------------------------------------------
        # SAFE COPY
        # ----------------------------------------------------

        dashboard_data = dataset.copy()

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        dashboard_data["date"] = pd.to_datetime(
            dashboard_data["date"],
            errors="coerce"
        )

        dashboard_data["date_only"] = (
            dashboard_data["date"]
            .dt.strftime("%Y-%m-%d")
        )

        # ----------------------------------------------------
        # TEC TREND
        # ----------------------------------------------------

        trend_data = (
            dashboard_data
            .dropna(
                subset=[
                    "date_only",
                    "tec"
                ]
            )
            .groupby("date_only")["tec"]
            .mean()
            .tail(30)
        )

        trend_labels = [
            str(value)
            for value in trend_data.index.tolist()
        ]

        trend_values = [
            round(float(value), 3)
            for value in trend_data.values.tolist()
        ]

        # ----------------------------------------------------
        # TEC DISTRIBUTION
        # ----------------------------------------------------

        low_count = int(
            (dashboard_data["tec"] < 5).sum()
        )

        medium_count = int(
            (
                (dashboard_data["tec"] >= 5)
                &
                (dashboard_data["tec"] < 10)
            ).sum()
        )

        high_count = int(
            (dashboard_data["tec"] >= 10).sum()
        )

        distribution_labels = [
            "Low TEC (< 5)",
            "Medium TEC (5 - 10)",
            "High TEC (>= 10)"
        ]

        distribution_values = [
            low_count,
            medium_count,
            high_count
        ]

        # ----------------------------------------------------
        # PREDICTION HISTORY
        # ----------------------------------------------------

        predictions = get_prediction_history(
            limit=20
        )

        prediction_count = get_prediction_count()

        # ----------------------------------------------------
        # RENDER DASHBOARD
        # ----------------------------------------------------

        return render_template(

            "dashboard.html",

            total_records=len(
                dashboard_data
            ),

            avg_tec=round(
                float(
                    dashboard_data["tec"].mean()
                ),
                3
            ),

            max_tec=round(
                float(
                    dashboard_data["tec"].max()
                ),
                3
            ),

            prediction_count=prediction_count,

            predictions=predictions,

            trend_labels=trend_labels,

            trend_values=trend_values,

            distribution_labels=distribution_labels,

            distribution_values=distribution_values

        )

    except Exception as error:

        print("\n" + "=" * 70)
        print("DASHBOARD ERROR")
        print("=" * 70)

        print(
            traceback.format_exc()
        )

        return f"""
        <h1>Dashboard Loading Error</h1>
        <pre>{str(error)}</pre>
        """, 500


# ============================================================
# ABOUT PAGE
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict_tec():

    try:

        initialize_application()

        if (
            extra_trees_model is None
            or
            knn_model is None
            or
            scaler is None
        ):

            return jsonify({

                "success": False,

                "error":
                    "Prediction model is not loaded."

            }), 503

        data = request.get_json(
            force=True,
            silent=True
        )

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No JSON data received."

            }), 400

        # ----------------------------------------------------
        # READ INPUT
        # ----------------------------------------------------

        location = str(
            data.get(
                "location",
                ""
            )
        ).strip()

        latitude_value = data.get(
            "latitude"
        )

        longitude_value = data.get(
            "longitude"
        )

        date_string = str(
            data.get(
                "date",
                ""
            )
        ).strip()

        time_string = str(
            data.get(
                "time",
                ""
            )
        ).strip()

        # ----------------------------------------------------
        # REQUIRED INPUT VALIDATION
        # ----------------------------------------------------

        if latitude_value in (
            None,
            ""
        ):

            raise ValueError(
                "Latitude is required."
            )

        if longitude_value in (
            None,
            ""
        ):

            raise ValueError(
                "Longitude is required."
            )

        latitude = float(
            latitude_value
        )

        longitude = float(
            longitude_value
        )

        if not location:

            location = (
                "Unknown Location"
            )

        if not (
            -90 <= latitude <= 90
        ):

            raise ValueError(
                "Latitude must be between -90 and 90."
            )

        if not (
            -180 <= longitude <= 180
        ):

            raise ValueError(
                "Longitude must be between -180 and 180."
            )

        if not date_string:

            raise ValueError(
                "Date is required."
            )

        if not time_string:

            raise ValueError(
                "Time is required."
            )

        # ----------------------------------------------------
        # VALIDATE DATE AND TIME FORMAT
        # ----------------------------------------------------

        datetime.strptime(
            f"{date_string} {time_string}",
            "%Y-%m-%d %H:%M"
        )

        # ----------------------------------------------------
        # CREATE FEATURES
        # ----------------------------------------------------

        full_input, knn_input = (
            create_prediction_features(

                latitude,

                longitude,

                date_string,

                time_string

            )
        )

        # ----------------------------------------------------
        # EXTRA TREES PREDICTION
        # ----------------------------------------------------

        extra_trees_prediction = float(

            extra_trees_model.predict(
                full_input
            )[0]

        )

        # ----------------------------------------------------
        # KNN PREDICTION
        # ----------------------------------------------------

        knn_scaled = scaler.transform(
            knn_input
        )

        knn_prediction = float(

            knn_model.predict(
                knn_scaled
            )[0]

        )

        # ----------------------------------------------------
        # HYBRID PREDICTION
        # ----------------------------------------------------

        predicted_tec = (

            0.65
            *
            extra_trees_prediction

            +

            0.35
            *
            knn_prediction

        )

        if not np.isfinite(
            predicted_tec
        ):

            raise ValueError(
                "Invalid prediction generated."
            )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        activity, risk = (
            classify_ionosphere(
                predicted_tec
            )
        )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        save_prediction_history(

            latitude,

            longitude,

            location,

            date_string,

            time_string,

            predicted_tec,

            activity,

            risk

        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "predicted_tec":
                round(
                    float(predicted_tec),
                    4
                ),

            "latitude":
                latitude,

            "longitude":
                longitude,

            "location":
                location,

            "date":
                date_string,

            "time":
                time_string,

            "activity":
                activity,

            "risk":
                risk,

            "model":
                "ExtraTrees + Distance KNN",

            "model_mae":
                round(
                    float(model_mae),
                    4
                )

        }), 200

    except ValueError as error:

        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 400

    except Exception as error:

        print("\n" + "=" * 70)
        print("PREDICTION ERROR")
        print("=" * 70)

        print(
            traceback.format_exc()
        )

        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# DASHBOARD API
# ============================================================

@app.route("/api/dashboard")
def api_dashboard():

    try:

        initialize_application()

        if dataset is None:

            raise RuntimeError(
                "Dataset not loaded."
            )

        prediction_count = get_prediction_count()

        min_date = None
        max_date = None

        if dataset_min_date is not None:

            min_date = str(
                dataset_min_date.date()
            )

        if dataset_max_date is not None:

            max_date = str(
                dataset_max_date.date()
            )

        return jsonify({

            "success": True,

            "records":
                int(len(dataset)),

            "tec_min":
                round(
                    float(
                        dataset["tec"].min()
                    ),
                    3
                ),

            "tec_max":
                round(
                    float(
                        dataset["tec"].max()
                    ),
                    3
                ),

            "tec_average":
                round(
                    float(
                        dataset["tec"].mean()
                    ),
                    3
                ),

            "prediction_count":
                prediction_count,

            "model_mae":
                round(
                    float(
                        model_mae or 0
                    ),
                    4
                ),

            "min_date":
                min_date,

            "max_date":
                max_date

        })

    except Exception as error:

        print(
            traceback.format_exc()
        )

        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# STATISTICS API
# ============================================================

@app.route("/api/statistics")
def statistics():

    try:

        initialize_application()

        if dataset is None:

            raise RuntimeError(
                "Dataset not loaded."
            )

        return jsonify({

            "success": True,

            "records":
                int(len(dataset)),

            "minimum":
                float(
                    dataset["tec"].min()
                ),

            "maximum":
                float(
                    dataset["tec"].max()
                ),

            "average":
                float(
                    dataset["tec"].mean()
                ),

            "model_mae":
                float(
                    model_mae or 0
                )

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# PREDICTION HISTORY API
# ============================================================

@app.route("/api/history")
def api_history():

    try:

        initialize_application()

        history = get_prediction_history(
            limit=50
        )

        return jsonify({

            "success": True,

            "count":
                len(history),

            "history":
                history

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    try:

        initialize_application()

        return jsonify({

            "status":
                "healthy",

            "application":
                "Ionosphere-AI",

            "database":
                "READY",

            "data_engine":
                "READY",

            "model":
                "MODEL READY"

        })

    except Exception as error:

        return jsonify({

            "status":
                "unhealthy",

            "error":
                str(error)

        }), 500


# ============================================================
# INITIALIZE APPLICATION
# IMPORTANT FOR RENDER / GUNICORN DEPLOYMENT
# ============================================================

try:

    initialize_application()

except Exception:

    print("\n" + "=" * 70)
    print("APPLICATION INITIALIZATION ERROR")
    print("=" * 70)

    print(
        traceback.format_exc()
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)

    print(
        "IONOSPHERE-AI TEC PREDICTION & ANALYTICS DASHBOARD"
    )

    print("=" * 70)

    if dataset is not None:

        print(
            f"Records: {len(dataset)}"
        )

    if model_mae is not None:

        print(
            f"Model MAE: {model_mae:.4f}"
        )

    print(
        "Prediction History: SQLite READY"
    )

    print(
        "Server: http://127.0.0.1:5050"
    )

    print("=" * 70 + "\n")

    app.run(

        debug=False,

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                5050
            )
        ),

        threaded=True,

        use_reloader=False

    )
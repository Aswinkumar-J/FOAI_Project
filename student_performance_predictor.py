import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import render_template, send_file, request
import io

# ML & preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Flask for serving
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except Exception:
    FLASK_AVAILABLE = False

# ---------------------
# Configuration & sample columns
# ---------------------
SAMPLE_COLUMNS = [
    "student_id",        # optional unique id
    "gender",            # categorical
    "age",               # numeric
    "parent_education",  # categorical
    "family_income",     # numeric (or categorical bucket)
    "attendance_pct",    # numeric 0-100
    "internal_marks",    # numeric 0-100 (average of internal assessments)
    "assignments_submitted_pct", # numeric 0-100
    "previous_grade",    # categorical (A/B/C/D/F) or numeric GPA
    "extracurricular",   # categorical yes/no/low/med/high
    # target:
    "final_result"       # categorical: 'Pass' or 'Fail' or multi-class like 'A','B','C'
]

TARGET_COLUMN = "final_result"

# ---------------------
# Utilities
# ---------------------
def generate_synthetic_data(n=2000, filename=None, seed=None):
    """
    Generate a realistic synthetic dataset for demo/training.
    If seed is None, uses a random seed so each run is unique.
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()  # system time-based random seed

    genders = ['Male', 'Female', 'Other']
    parent_ed = ['None', 'Primary', 'Secondary', 'Graduate', 'Postgraduate']
    prev_grade = ['A', 'B', 'C', 'D', 'F']
    exc = ['None', 'Low', 'Medium', 'High']

    rows = []
    for i in range(n):
        gender = np.random.choice(genders, p=[0.47, 0.47, 0.06])
        age = np.random.randint(16, 23)
        ped = np.random.choice(parent_ed, p=[0.05,0.15,0.4,0.3,0.1])
        income = int(np.random.normal(25000, 15000))
        income = max(2000, income)
        attendance = np.clip(np.random.normal(80 + (0 if ped in ['None','Primary'] else 5), 10), 40, 100)
        internal = np.clip(np.random.normal(65 + (5 if ped in ['Graduate','Postgraduate'] else 0), 15), 20, 100)
        assignments = np.clip(np.random.normal(attendance - 5, 12), 10, 100)
        prev = np.random.choice(prev_grade, p=[0.15,0.25,0.3,0.2,0.1])
        extracurricular = np.random.choice(exc, p=[0.2,0.5,0.2,0.1])

        score = (attendance * 0.25 + internal * 0.4 + assignments * 0.15
                 + (10 if prev in ['A','B'] else -5 if prev in ['D','F'] else 0)
                 + (5 if extracurricular in ['Medium','High'] else 0)
                 + (5 if ped in ['Graduate','Postgraduate'] else 0)
                )
        pass_prob = 1 / (1 + np.exp(-(score - 60)/7.0))
        final = 'Pass' if np.random.rand() < pass_prob else 'Fail'

        rows.append({
            "student_id": f"S{i+1:05d}",
            "gender": gender,
            "age": age,
            "parent_education": ped,
            "family_income": income,
            "attendance_pct": round(attendance,1),
            "internal_marks": round(internal,1),
            "assignments_submitted_pct": round(assignments,1),
            "previous_grade": prev,
            "extracurricular": extracurricular,
            "final_result": final
        })

    df = pd.DataFrame(rows)
    if filename:
        df.to_csv(filename, index=False)
        print(f"[INFO] Synthetic sample saved to {filename}")
    return df

# ---------------------
# Preprocessing pipeline builder
# ---------------------
def build_preprocessing_pipeline(df: pd.DataFrame, target_col=TARGET_COLUMN):
    """Detect columns automatically and construct ColumnTransformer"""
    # Basic heuristics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude student_id-like and target
    numeric_cols = [c for c in numeric_cols if c not in (target_col,)]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target_col]]
    # Remove obvious ID columns
    categorical_cols = [c for c in categorical_cols if 'id' not in c.lower()]

    # define transformers
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, categorical_cols)
    ], remainder='drop', verbose_feature_names_out=False)

    return preprocessor, numeric_cols, categorical_cols
# ---------------------
# Model builders
# ---------------------
def build_rf_model(random_state=42):
    return RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=random_state)

def build_xgb_model(random_state=42):
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost not available. Install xgboost package.")
    return xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=-1)

def build_keras_model(input_dim, dropout_rate=0.2):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Install tensorflow package.")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate/2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # binary classification (Pass/Fail)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---------------------
# Training orchestration
# ---------------------
def train_and_evaluate(df: pd.DataFrame, target_col=TARGET_COLUMN, save_dir="saved", random_state=42):
    os.makedirs(save_dir, exist_ok=True)

    # Convert target to binary numeric
    df = df.copy()
    # if multi-class, for demo we use Pass/Fail mapping
    df[target_col] = df[target_col].astype(str)
    classes = sorted(df[target_col].unique().tolist())
    if set(classes) == set(['Pass','Fail']) or 'Pass' in classes:
        df[target_col] = df[target_col].apply(lambda x: 1 if x=='Pass' else 0)
    else:
        # fallback: label encode
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        joblib.dump(le, os.path.join(save_dir, 'label_encoder.pkl'))
        print("[INFO] Saved label encoder.")

    X = df.drop(columns=[target_col, 'student_id'] if 'student_id' in df.columns else [target_col])
    y = df[target_col].values

    preprocessor, numeric_cols, categorical_cols = build_preprocessing_pipeline(pd.concat([X, df[[target_col]]], axis=1), target_col)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)

    # Build base estimators
    rf = build_rf_model(random_state)
    models = []

    # Random forest pipeline
    rf_pipeline = Pipeline(steps=[('pre', preprocessor), ('clf', rf)])
    models.append(('rf', rf_pipeline))

    # XGBoost pipeline
    if XGBOOST_AVAILABLE:
        xgb_clf = build_xgb_model(random_state)
        xgb_pipeline = Pipeline(steps=[('pre', preprocessor), ('clf', xgb_clf)])
        models.append(('xgb', xgb_pipeline))
    else:
        print("[WARN] XGBoost not installed — skipping xgb model.")

    # Keras model as scikit-learn wrapper
    if TF_AVAILABLE:
        # Need to know processed feature dimension
        X_prep = preprocessor.fit_transform(X_train)
        input_dim = X_prep.shape[1]
        keras_wrapper = KerasClassifier(build_fn=lambda: build_keras_model(input_dim), epochs=30, batch_size=32, verbose=0)
        keras_pipeline = Pipeline(steps=[('pre', preprocessor), ('clf', keras_wrapper)])
        models.append(('keras', keras_pipeline))
    else:
        print("[WARN] TensorFlow not installed — skipping Keras model.")

    # Create a strong ensemble (Voting) using available models
    voting = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
    # Calibrate to produce well-calibrated probabilities
    calib = CalibratedClassifierCV(estimator=voting, cv=3, method='isotonic')  # can change to 'sigmoid' if data small
    print("[INFO] Training ensemble (this may take a while)...")
    calib.fit(X_train, y_train)

    # Evaluate
    y_pred = calib.predict(X_test)
    y_proba = calib.predict_proba(X_test)[:,1]
    print("\n[RESULTS] Evaluation on hold-out test set:")
    print(classification_report(y_test, y_pred, digits=4))
    try:
        roc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {roc:.4f}")
    except Exception:
        pass

    # Save pipeline
    model_path = os.path.join(save_dir, 'pipeline.pkl')
    joblib.dump({'model': calib, 'preprocessor_cols': {'numeric': numeric_cols, 'categorical': categorical_cols}}, model_path)
    print(f"[INFO] Saved trained pipeline to {model_path}")

    # Optional: SHAP explainability for the ensemble's predictions (if shap installed)
    if SHAP_AVAILABLE:
        print("[INFO] Generating SHAP explainer using TreeExplainer where possible...")
        # Attempt to extract a tree-based estimator for background; fallback to KernelExplainer (slow)
        try:
            # get preprocessed X_train for explainer
            X_train_prep = preprocessor.transform(X_train)
            # try TreeExplainer on RF inside the ensemble if available
            # Find first estimator that's tree-like
            tree_est = None
            for name, est in calib.base_estimator_.estimators:
                clf = est.named_steps['clf'] if isinstance(est, Pipeline) else est
                if hasattr(clf, 'estimators_') or 'XGBClassifier' in str(type(clf)) or 'RandomForest' in str(type(clf)):
                    tree_est = clf
                    break
            if tree_est is not None:
                explainer = shap.Explainer(tree_est, X_train_prep)
                shap_values = explainer(X_test)  # this can be large
                # Save a small summary
                shap.summary_plot(shap_values, features=preprocessor.transform(X_test), show=False)
                joblib.dump({'shap_explainer': explainer}, os.path.join(save_dir, 'shap_explainer.pkl'))
                print("[INFO] SHAP explainer saved.")
            else:
                print("[WARN] No tree estimator found for fast SHAP. Skipping SHAP.")
        except Exception as e:
            print(f"[WARN] SHAP analysis failed: {e}")

    return os.path.abspath(model_path)

# def predict_single_simple(model_artifact, payload: dict):
#     """
#     Simplified single student predictor:
#     Uses attendance, internal_marks, assignments_submitted_pct, previous_grade, extracurricular only.
#     Reduces influence of internal_marks, increases previous_grade influence.
#     """
#     model = model_artifact['model']
#     remarks = []

#     # Hard validation
#     limits = {
#         'attendance_pct': (0, 100),
#         'internal_marks': (0, 100),
#         'assignments_submitted_pct': (0, 100),
#     }

#     for field, (min_val, max_val) in limits.items():
#         if field in payload:
#             try:
#                 val = float(payload[field])
#                 payload[field] = min(max(val, min_val), max_val)
#             except (TypeError, ValueError):
#                 payload[field] = min_val

#     # Drop non-used features if present
#     for unused in ['gender', 'age', 'parent_education', 'family_income', 'student_id']:
#         payload.pop(unused, None)

#     df = pd.DataFrame([payload])

#     # Base model probability
#     try:
#         proba = model.predict_proba(df)[:, 1][0]
#     except Exception:
#         proba = float(model.predict(df)[0])

#     # Adjust probability manually
#     # Internal marks influence reduced
#     internal = payload.get('internal_marks', 70)
#     internal_adj = (internal - 50) * 0.005  # smaller weight

#     # Previous grade influence increased
#     prev_grade = str(payload.get('previous_grade', 'C')).upper()
#     grade_weights = {'A': 0.03, 'B': 0.02, 'C': 0.0, 'D': -0.02, 'F': -0.03}
#     grade_adj = grade_weights.get(prev_grade, 0.0)

#     # Assignments influence small
#     assign = payload.get('assignments_submitted_pct', 80)
#     assign_adj = (assign - 70) * 0.002

#     # Extracurricular reverse influence
#     extra_map = {'None': 0.02, 'Low': 0.01, 'Medium': -0.01, 'High': -0.02}
#     extra_adj = extra_map.get(str(payload.get('extracurricular', 'Medium')), 0.0)

#     # Combine adjustments
#     proba += internal_adj + grade_adj + assign_adj + extra_adj
#     proba = min(max(proba, 0.0), 1.0)

#     # Remarks
#     if payload.get('attendance_pct', 100) < 75:
#         remarks.append("Low attendance may affect performance")
#     if internal < 50:
#         remarks.append("Internal marks are low")
#     if payload.get('assignments_submitted_pct', 100) < 70:
#         remarks.append("Assignments submission is low")
#     if prev_grade in ['D','F']:
#         remarks.append("Previous grade is low")
#     if not remarks:
#         remarks.append("Performance appears consistent across key factors")

#     pred_label = 1 if proba >= 0.5 else 0

#     return {
#         'prediction': int(pred_label),
#         'probability': round(proba * 100, 2),
#         'remarks': "; ".join(remarks)
#     }

def predict_single_simple(model_artifact, payload: dict):
    """
    Simplified prediction with consistent grade ordering per unique student profile:
    Remembers previous probabilities for the same (attendance, internal_marks, assignments_submitted_pct) combination.
    """
    import pandas as pd
    import hashlib
    global grade_memory

    # Initialize memory if not present
    if 'grade_memory' not in globals():
        grade_memory = {}

    model = model_artifact['model']

    # Add placeholders for ignored columns
    for col, val in {
        'gender': 'Male',
        'age': 18,
        'parent_education': 'Secondary',
        'family_income': 25000
    }.items():
        payload.setdefault(col, val)

    # Validate numeric features
    limits = {'attendance_pct': (0, 100), 'internal_marks': (0, 100), 'assignments_submitted_pct': (0, 100)}
    for field, (min_val, max_val) in limits.items():
        try:
            payload[field] = min(max(float(payload.get(field, 0)), min_val), max_val)
        except:
            payload[field] = min_val

    payload.pop('student_id', None)

    # Create unique key for this specific student profile (excluding grade)
    profile_key_raw = f"{payload.get('attendance_pct')}_{payload.get('internal_marks')}_{payload.get('assignments_submitted_pct')}_{payload.get('extracurricular','Low')}"
    profile_key = hashlib.md5(profile_key_raw.encode()).hexdigest()

    # Initialize memory for this profile
    if profile_key not in grade_memory:
        grade_memory[profile_key] = {"A": None, "B": None, "C": None, "D": None, "F": None}

    model_cache = grade_memory[profile_key]

    # Base prediction from model
    df = pd.DataFrame([payload])
    try:
        raw_proba = model.predict_proba(df)[:, 1]
    except Exception:
        raw_proba = model.predict(df)
    proba = float(raw_proba[0])

    # Adjust manually
    internal_effect = (payload.get('internal_marks', 50) / 100) * 0.08
    grade_addition = {"A": 0.22, "B": 0.14, "C": 0.08, "D": 0.04, "F": 0.0}
    prev_grade = str(payload.get('previous_grade', 'C')).upper().strip()
    exc_weights = {'High': -0.06, 'Medium': -0.05, 'Low': -0.02, 'None': 0.0}
    exc_level = str(payload.get('extracurricular', 'Low')).capitalize()

    # Weighted combination
    proba = proba * 0.6 + internal_effect + grade_addition.get(prev_grade, 0.08) + exc_weights.get(exc_level, 0.0)
    proba = min(max(proba, 0.0), 1.0)

    # --- Maintain consistent ordering for this specific profile ---
    grade_order = ["A", "B", "C", "D", "F"]
    model_cache[prev_grade] = proba

    # Ensure A > B > C > D > F with 3% difference
    for i in range(len(grade_order) - 1):
        g_high, g_low = grade_order[i], grade_order[i + 1]
        if model_cache[g_high] is not None and model_cache[g_low] is not None:
            if model_cache[g_high] <= model_cache[g_low]:
                model_cache[g_high] = model_cache[g_low] + 0.03

    # Save updated profile
    grade_memory[profile_key] = model_cache
    proba = model_cache[prev_grade]
    proba = min(max(proba, 0.0), 1.0)

    # Remarks
    remarks = []
    if payload.get('attendance_pct', 100) < 75:
        remarks.append("Low attendance may affect performance")
    if payload.get('assignments_submitted_pct', 100) < 70:
        remarks.append("Assignments submission is low")
    if payload.get('internal_marks', 100) < 50:
        remarks.append("Internal marks are low")
    if prev_grade in ['D', 'F']:
        remarks.append("Previous grade is low")
    if not remarks:
        remarks.append("No issues")

    return {
        'prediction': int(proba >= 0.5),
        'probability': round(proba * 100, 2),
        'remarks': remarks
    }



# def predict_single_simple(model_artifact, payload: dict):
#     """
#     Simplified prediction with:
#     - Internal marks: small effect
#     - Previous grade: larger A->B gap
#     - Extracurricular: strictly decreasing effect
#     - Max probability near 98%
#     """
#     model = model_artifact['model']

#     # --- Add placeholders for ignored columns ---
#     for col, val in {
#         'gender': 'Male',
#         'age': 18,
#         'parent_education': 'Secondary',
#         'family_income': 25000
#     }.items():
#         if col not in payload:
#             payload[col] = val

#     # --- Validate numeric features ---
#     limits = {
#         'attendance_pct': (0, 100),
#         'internal_marks': (0, 100),
#         'assignments_submitted_pct': (0, 100)
#     }
#     for field, (min_val, max_val) in limits.items():
#         if field in payload:
#             try:
#                 val = float(payload[field])
#                 payload[field] = min(max(val, min_val), max_val)
#             except (TypeError, ValueError):
#                 payload[field] = min_val

#     payload.pop('student_id', None)

#     # --- Base prediction from model ---
#     import pandas as pd
#     df = pd.DataFrame([payload])
#     try:
#         raw_proba = model.predict_proba(df)[:, 1]
#     except Exception:
#         raw_proba = model.predict(df)
#     proba = float(raw_proba[0])

#     # --- Manual weighting adjustments ---
#     # Internal marks: small contribution
#     internal_effect = (payload.get('internal_marks', 50) / 100) * 0.08

#     # Previous grade: distinct A-B-C gaps
#     grade_addition = {"A": 0.22, "B": 0.14, "C": 0.08, "D": 0.04, "F": 0.0}
#     prev_grade = str(payload.get('previous_grade', 'C')).upper().strip()

#     # Weighted combination: grade stronger, model moderated
#     proba = proba * 0.6 + internal_effect + grade_addition.get(prev_grade, 0.08)

#     # --- Extracurricular effect (strictly decreasing) ---
#     exc_level = str(payload.get('extracurricular', 'Low')).strip().lower()
#     exc_penalty = {
#         "none": +0.03,   # highest probability
#         "low": 0.00,
#         "medium": -0.06,  # slightly lower
#         "high": -0.12     # clearly lowest
#     }
#     proba += exc_penalty.get(exc_level, 0.00)

#     # --- Clamp probability to [0, 1] ---
#     proba = min(max(proba, 0.0), 1.0)

#     # --- Remarks ---
#     remarks = []
#     if payload.get('attendance_pct', 100) < 75:
#         remarks.append("Low attendance may affect performance")
#     if payload.get('assignments_submitted_pct', 100) < 70:
#         remarks.append("Assignments submission is low")
#     if payload.get('internal_marks', 100) < 50:
#         remarks.append("Internal marks are low")
#     if prev_grade in ['D', 'F']:
#         remarks.append("Previous grade is low")

#     if not remarks:
#         remarks = ["No issues detected"]

#     return {
#         'prediction': int(proba >= 0.5),
#         'probability': round(proba * 100, 2),
#         'remarks': remarks
#     }







# ---------------------
# Simple predict function to be used by Flask or CLI
# ---------------------
def predict_single(model_artifact, payload: dict):
    """
    Predict result using deterministic weighting.
    Ensures lower inputs reduce probability, and extracurricular adds small effect.
    """
    remarks = []

    # Hard numeric bounds
    numeric_fields = {
        'age': (4, 100),
        'attendance_pct': (0, 100),
        'internal_marks': (0, 100),
        'assignments_submitted_pct': (0, 100),
        'family_income': (0, 1_000_000)
    }
    for k, (lo, hi) in numeric_fields.items():
        if k in payload:
            val = payload[k]
            try:
                val = float(val)
                val = max(lo, min(val, hi))
            except:
                val = lo
            payload[k] = val

    # -----------------------
    # Deterministic probability calculation
    # -----------------------
    # Base score from main numeric features
    score = 0.4 * payload.get('internal_marks', 0) + 0.3 * payload.get('attendance_pct', 0) + 0.2 * payload.get('assignments_submitted_pct', 0)

    # Previous grade weight
    grade_weights = {'A': 10, 'B': 5, 'C': 0, 'D': -5, 'F': -10}
    # grade_weights = {'A': 10, 'B': 0, 'C': -5, 'D': -10, 'F': -15}
    prev_grade = str(payload.get('previous_grade', 'C')).upper().strip()
    score += grade_weights.get(prev_grade, 0)

    # Small weight for extracurricular involvement (inverse effect)
    exc_weights = {'High': -5, 'Medium': -2, 'Low': 0, 'None': 2}
    exc_level = str(payload.get('extracurricular', 'Low')).capitalize()
    score += exc_weights.get(exc_level, 0)

    # Normalize to 0–1
    proba = 1 / (1 + np.exp(-(score - 60)/10.0))

    # -----------------------
    # Remarks
    # -----------------------
    if payload.get('attendance_pct', 100) < 75:
        remarks.append("Low attendance may affect performance")
    if payload.get('assignments_submitted_pct', 100) < 70:
        remarks.append("Assignments submission is low")
    if payload.get('internal_marks', 100) < 50:
        remarks.append("Internal marks are low")
    if prev_grade in ['D','F']:
        remarks.append("Previous grade is low")
    if payload.get('age', 18) < 6:
        remarks.append("Very young student")
    if exc_level in ['Medium', 'High']:
        remarks.append(f"Good extracurricular involvement ({exc_level})")

    # If no remarks, default to "No issues"
    if not remarks:
        remarks = ["No issues"]

    # Prediction threshold 0.5
    prediction = 1 if proba >= 0.5 else 0

    return {
        'prediction': prediction,
        'probability': round(proba * 100, 2),
        'remarks': remarks
    }





# ---------------------
# Minimal Flask server
# ---------------------
def run_flask(model_path, host='0.0.0.0', port=5000):
    if not FLASK_AVAILABLE:
        raise RuntimeError("Flask not installed. Install flask to use server mode.")
    
    # Load model **once** at app startup
    artifact = joblib.load(model_path)
    model = artifact['model']

    app = Flask(__name__)

    # Route to download CSV

    @app.route('/predict_simple', methods=['GET', 'POST'])
    def predict_simple_route():
        if request.method == 'GET':
            return render_template('predict_simple.html')  # your HTML file
        else:
            try:
                payload = request.get_json()
                if isinstance(payload, list):
                    results = []
                    for p in payload:
                        res = predict_single_simple(artifact, p)
                        results.append(res)
                    return jsonify({'predictions': results})
                else:
                    res = predict_single_simple(artifact, payload)
                    return jsonify(res)
            except Exception as e:
                return jsonify({'error': str(e)}), 400




    @app.route('/download_csv')
    def download_csv():
        csv_string = request.args.get('csv')
        if not csv_string:
            return "No data to download", 400
        return send_file(
            io.BytesIO(csv_string.encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='batch_predictions.csv'
        )



    @app.route('/ping', methods=['GET'])
    def ping():
        return jsonify({'status': 'ok', 'time': str(datetime.utcnow())})

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            payload = request.get_json()
            if isinstance(payload, list):
                results = []
                for p in payload:
                    res = predict_single(artifact, p)
                    results.append(res)
                return jsonify({'predictions': results})
            else:
                res = predict_single(artifact, payload)
                return jsonify(res)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # Batch prediction route
    @app.route('/batch', methods=['GET', 'POST'])
    def batch_predict():
        if request.method == 'GET':
            return render_template('batch.html')

        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            # Read file
            if file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)

            # Required columns
            required_cols = ['gender', 'age', 'parent_education', 'family_income',
                            'attendance_pct', 'internal_marks', 'assignments_submitted_pct',
                            'previous_grade', 'extracurricular']
            for c in required_cols:
                if c not in df.columns:
                    return jsonify({'error': f'Missing required column: {c}'}), 400

            # Validate numeric fields
            limits = {
                'age': (4, 100),
                'family_income': (0, 1_000_000),
                'attendance_pct': (0, 100),
                'internal_marks': (0, 100),
                'assignments_submitted_pct': (0, 100),
            }
            for field, (min_val, max_val) in limits.items():
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
                    df[field] = df[field].fillna(min_val)
                    df[field] = df[field].clip(lower=min_val, upper=max_val)

            # Predict once per session
            model = artifact['model']
            preds = model.predict(df)
            probs = model.predict_proba(df)[:, 1]

            df['Prediction'] = ['Pass' if p == 1 else 'Fail' for p in preds]
            df['Probability'] = (probs * 100).round(2)

            # Generate remarks
            remarks_list = []
            for _, row in df.iterrows():
                remarks = []
                if row['attendance_pct'] < 75:
                    remarks.append("Low attendance may affect performance")
                if row['assignments_submitted_pct'] < 70:
                    remarks.append("Assignments submission is low")
                if row['internal_marks'] < 50:
                    remarks.append("Internal marks are low")
                if row['age'] < 6:
                    remarks.append("Very young student")
                remarks_list.append("; ".join(remarks) if remarks else "No issues")
            df['Remarks'] = remarks_list

            # HTML table
            cols_to_show = ['Prediction', 'Probability', 'Remarks']
            if 'student_id' in df.columns:
                cols_to_show = ['student_id'] + cols_to_show
            html_table = df[cols_to_show].to_html(classes='table table-striped', index=False)

            # CSV buffer for download
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            csv_string = csv_buffer.getvalue()

            return render_template('batch.html', table=html_table, csv_string=csv_string)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    print(f"[INFO] Serving model from {model_path} at http://{host}:{port}")
    app.run(host=host, port=port)

# ---------------------
# CLI
# ---------------------
def main():
    parser = argparse.ArgumentParser(description='Student Performance Predictor')
    parser.add_argument('--generate-sample', action='store_true', help='Generate a synthetic sample CSV')
    parser.add_argument('--out', type=str, default='sample_students.csv', help='Output CSV for sample generation')
    parser.add_argument('--data', type=str, help='CSV data file for training')
    parser.add_argument('--train', action='store_true', help='Train model on provided CSV')
    parser.add_argument('--serve', action='store_true', help='Start Flask server with saved model')
    parser.add_argument('--model', type=str, default='saved/pipeline.pkl', help='Path to trained pipeline (for serving)')
    parser.add_argument('--nrows', type=int, default=None, help='Limit rows loaded from CSV (for quick tests)')
    args = parser.parse_args()

    if args.generate_sample:
        generate_synthetic_data(n=2000, filename=args.out)
        return

    if args.train:
        if not args.data:
            raise ValueError("Please provide --data <csvfile> to train.")
        print(f"[INFO] Loading data from {args.data} ...")
        df = pd.read_csv(args.data, nrows=args.nrows)
        print(f"[INFO] Loaded {len(df)} rows and columns: {list(df.columns)}")
        model_path = train_and_evaluate(df)
        print(f"[INFO] Training finished. Model saved to: {model_path}")
        return

    if args.serve:
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model}")
        run_flask(args.model)
        return

    parser.print_help()

if __name__ == '__main__':
    main()

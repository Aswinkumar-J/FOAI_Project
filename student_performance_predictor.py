"""
student_performance_predictor.py
A complete pipeline to train, explain, and serve a Student Performance Predictor.

Usage:
    1) Place a CSV with student data (see SAMPLE_COLUMNS below) or generate synthetic data:
         python student_performance_predictor.py --generate-sample --out sample_students.csv
    2) Train:
         python student_performance_predictor.py --train --data sample_students.csv
    3) Serve:
         python student_performance_predictor.py --serve --model saved/pipeline.pkl

Author: Generated for your Student Performance Predictor project.
"""

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
def generate_synthetic_data(n=2000, filename=None, seed=42):
    """Generate a realistic synthetic dataset for demo/training."""
    np.random.seed(seed)
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

        # Create a probability of passing influenced by features
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

# ---------------------
# Simple predict function to be used by Flask or CLI
# ---------------------
def predict_single(model_artifact, payload: dict):
    """
    Validate, predict single student's result, and generate remarks.
    """
    model = model_artifact['model']

    # -------- Validation --------
    # Set safe bounds for numeric features
    limits = {
        'age': (4, 100),
        'family_income': (0, 1_000_000),
        'attendance_pct': (0, 100),
        'internal_marks': (0, 100),
        'assignments_submitted_pct': (0, 100),
    }

    for field, (min_val, max_val) in limits.items():
        if field in payload:
            try:
                val = float(payload[field])
                if val < min_val:
                    payload[field] = min_val
                elif val > max_val:
                    payload[field] = max_val
            except (TypeError, ValueError):
                payload[field] = min_val

    # Drop student_id if present
    if 'student_id' in payload:
        payload.pop('student_id')

    # -------- Prediction --------
    df = pd.DataFrame([payload])
    proba = model.predict_proba(df)[:, 1] if hasattr(model, 'predict_proba') else model.predict(df)
    pred_label = model.predict(df)

    # -------- Generate remarks --------
    remarks = []
    if 'attendance_pct' in payload and payload['attendance_pct'] < 75:
        remarks.append("Low attendance may affect performance")
    if 'internal_marks' in payload and payload['internal_marks'] < 50:
        remarks.append("Internal marks are low")
    if 'assignments_submitted_pct' in payload and payload['assignments_submitted_pct'] < 70:
        remarks.append("Assignments submission is low")
    if 'age' in payload and payload['age'] < 10:
        remarks.append("Age is very low, may need review")
    
    if not remarks:
        remarks.append("Good standing")

    return {
        'prediction': int(pred_label[0]),
        'probability': float(proba[0]),
        'remarks': "; ".join(remarks)
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
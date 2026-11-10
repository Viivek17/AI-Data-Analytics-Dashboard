from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import pandas as pd
import numpy as np
import os, json, io
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-development-secret")
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def current_path():
    name = session.get("dataset")
    if not name:
        return None
    path = UPLOAD_DIR / name
    return path if path.exists() else None

def read_dataset(path=None):
    path = path or current_path()
    if not path:
        return None
    suffix = Path(path).suffix.lower()
    try:
        return pd.read_csv(path) if suffix == ".csv" else pd.read_excel(path)
    except Exception as exc:
        flash(f"Could not read dataset: {exc}", "error")
        return None

def safe_num(value):
    try:
        if pd.isna(value) or not np.isfinite(float(value)):
            return None
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None

def dataset_summary(df):
    numeric = df.select_dtypes(include=np.number)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": int(len(numeric.columns)),
        "categorical_columns": int(len(df.columns) - len(numeric.columns)),
        "memory": f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
    }

def column_quality(df):
    result = []
    for col in df.columns:
        result.append({
            "name": str(col),
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique(dropna=True)),
            "missing_pct": round(float(df[col].isna().mean() * 100), 1)
        })
    return result

def chart_data(df):
    numeric = df.select_dtypes(include=np.number)
    charts = {"numeric_labels": [], "numeric_values": [], "missing_labels": [], "missing_values": [],
              "category_labels": [], "category_values": [], "corr_labels": [], "corr_values": []}
    if not numeric.empty:
        means = numeric.mean(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0).head(8)
        charts["numeric_labels"] = [str(x) for x in means.index]
        charts["numeric_values"] = [safe_num(x) or 0 for x in means.values]
        corr = numeric.corr(numeric_only=True).fillna(0)
        cols = list(corr.columns[:6])
        charts["corr_labels"] = cols
        charts["corr_values"] = [[safe_num(v) or 0 for v in row] for row in corr.loc[cols, cols].values]
    miss = df.isna().sum().sort_values(ascending=False).head(8)
    charts["missing_labels"] = [str(x) for x in miss.index]
    charts["missing_values"] = [int(x) for x in miss.values]
    for col in df.select_dtypes(exclude=np.number).columns[:1]:
        counts = df[col].astype(str).value_counts().head(7)
        charts["category_labels"] = [str(x) for x in counts.index]
        charts["category_values"] = [int(x) for x in counts.values]
    return charts

def insights_for(df):
    insights = []
    if df.empty:
        return ["Your dataset is empty. Upload a dataset with rows to get insights."]
    missing = int(df.isna().sum().sum())
    dupes = int(df.duplicated().sum())
    if missing:
        insights.append(f"{missing:,} missing cell(s) found across {int((df.isna().sum() > 0).sum())} column(s).")
    else:
        insights.append("Great data completeness: no missing cells were found.")
    if dupes:
        insights.append(f"{dupes:,} duplicate row(s) may be removed during data cleaning.")
    else:
        insights.append("No duplicate rows were detected.")
    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        means = numeric.mean(numeric_only=True).dropna()
        if not means.empty:
            col = str(means.idxmax())
            insights.append(f"'{col}' has the highest average among numeric columns ({means.max():,.2f}).")
        if len(numeric.columns) >= 2:
            corr = numeric.corr(numeric_only=True)
            pairs = []
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i+1:]:
                    value = corr.loc[c1, c2]
                    if pd.notna(value):
                        pairs.append((abs(value), c1, c2, value))
            if pairs:
                _, a, b, val = max(pairs)
                insights.append(f"Strongest numeric relationship in this dataset: '{a}' and '{b}' (correlation {val:.2f}).")
    insights.append(f"The dataset contains {len(df):,} rows and {len(df.columns)} columns.")
    return insights

@app.context_processor
def inject_globals():
    return {"dataset_name": session.get("dataset"), "year": datetime.now().year}

@app.route("/")
def dashboard():
    df = read_dataset()
    summary = dataset_summary(df) if df is not None else {"rows": 0, "columns": 0, "missing": 0, "duplicates": 0, "numeric_columns": 0, "categorical_columns": 0, "memory": "—"}
    return render_template("dashboard.html", summary=summary, has_data=df is not None,
                           insights=insights_for(df)[:3] if df is not None else [])

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or not file.filename:
            flash("Choose a CSV or Excel file first.", "error")
            return redirect(url_for("upload"))
        if not allowed_file(file.filename):
            flash("Unsupported format. Please upload CSV, XLSX, or XLS.", "error")
            return redirect(url_for("upload"))
        filename = secure_filename(file.filename)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stored_name = f"{stamp}_{filename}"
        file.save(UPLOAD_DIR / stored_name)
        session["dataset"] = stored_name
        flash("Dataset uploaded successfully.", "success")
        return redirect(url_for("preview"))
    return render_template("upload.html")

@app.route("/preview")
def preview():
    df = read_dataset()
    if df is None:
        flash("Upload a dataset to preview it.", "info")
        return redirect(url_for("upload"))
    page = max(1, request.args.get("page", 1, type=int))
    page_size = 10
    start = (page - 1) * page_size
    rows = df.iloc[start:start + page_size].replace({np.nan: None}).to_dict(orient="records")
    return render_template("preview.html", summary=dataset_summary(df), columns=list(df.columns),
                           rows=rows, page=page, pages=max(1, (len(df) + page_size - 1)//page_size),
                           total=len(df))

@app.route("/quality")
def quality():
    df = read_dataset()
    if df is None:
        flash("Upload a dataset to run data quality analysis.", "info")
        return redirect(url_for("upload"))
    return render_template("quality.html", summary=dataset_summary(df), quality=column_quality(df),
                           charts=chart_data(df), insights=insights_for(df))

@app.route("/cleaning", methods=["GET", "POST"])
def cleaning():
    df = read_dataset()
    if df is None:
        flash("Upload a dataset before cleaning.", "info")
        return redirect(url_for("upload"))
    if request.method == "POST":
        method = request.form.get("missing_method", "mean")
        remove_duplicates = request.form.get("remove_duplicates") == "on"
        convert_types = request.form.get("convert_types") == "on"
        before_missing, before_duplicates = int(df.isna().sum().sum()), int(df.duplicated().sum())
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    if method == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif method == "zero":
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(df[col].mean())
                else:
                    fill = df[col].mode(dropna=True)
                    df[col] = df[col].fillna(fill.iloc[0] if not fill.empty else "Unknown")
        if remove_duplicates:
            df = df.drop_duplicates()
        if convert_types:
            for col in df.columns:
                if df[col].dtype == object:
                    converted = pd.to_numeric(df[col], errors="coerce")
                    if converted.notna().mean() > 0.85:
                        df[col] = converted
        path = current_path()
        if path.suffix.lower() == ".csv":
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False)
        flash(f"Cleaning complete: handled {before_missing} missing cell(s); " +
              (f"removed {before_duplicates} duplicate row(s)." if remove_duplicates else "duplicate rows were kept."), "success")
        return redirect(url_for("cleaning"))
    return render_template("cleaning.html", summary=dataset_summary(df))

@app.route("/eda")
def eda():
    df = read_dataset()
    if df is None:
        flash("Upload a dataset to explore it.", "info")
        return redirect(url_for("upload"))
    numeric = df.select_dtypes(include=np.number)
    stats = []
    for col in numeric.columns:
        stats.append({"name": str(col), "mean": safe_num(numeric[col].mean()),
                      "median": safe_num(numeric[col].median()), "std": safe_num(numeric[col].std()),
                      "min": safe_num(numeric[col].min()), "max": safe_num(numeric[col].max())})
    return render_template("eda.html", summary=dataset_summary(df), charts=chart_data(df), stats=stats)

@app.route("/insights")
def insights():
    df = read_dataset()
    if df is None:
        flash("Upload a dataset to generate insights.", "info")
        return redirect(url_for("upload"))
    return render_template("insights.html", insights=insights_for(df), summary=dataset_summary(df))

@app.route("/ask-ai", methods=["GET", "POST"])
def ask_ai():
    answer = None
    question = ""
    df = read_dataset()
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if df is None:
            answer = "Please upload a dataset first so I can answer questions about it."
        elif not question:
            answer = "Type a question about your dataset."
        else:
            q = question.lower()
            numeric = df.select_dtypes(include=np.number)
            if any(word in q for word in ["row", "record", "how many data"]):
                answer = f"Your dataset contains {len(df):,} rows and {len(df.columns)} columns."
            elif "missing" in q or "null" in q:
                answer = f"There are {int(df.isna().sum().sum()):,} missing cells in the dataset."
            elif "duplicate" in q:
                answer = f"I found {int(df.duplicated().sum()):,} duplicate rows."
            elif ("average" in q or "mean" in q) and not numeric.empty:
                means = numeric.mean().dropna()
                answer = "Column averages: " + "; ".join(f"{c}: {v:.2f}" for c, v in means.items())
            elif ("maximum" in q or "highest" in q or "max" in q) and not numeric.empty:
                answer = "Maximum values: " + "; ".join(f"{c}: {numeric[c].max()}" for c in numeric.columns)
            elif "column" in q:
                answer = "Columns: " + ", ".join(map(str, df.columns))
            else:
                answer = "Here is a quick dataset overview: " + f"{len(df)} rows, {len(df.columns)} columns, {int(df.isna().sum().sum())} missing cells, and {int(df.duplicated().sum())} duplicate rows. Try asking about missing values, duplicates, averages, maximums, rows, or columns."
    return render_template("ask_ai.html", answer=answer, question=question, has_data=df is not None)

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    df = read_dataset()
    result = None
    numeric_cols = list(df.select_dtypes(include=np.number).columns) if df is not None else []
    if request.method == "POST":
        target = request.form.get("target")
        if df is None or target not in numeric_cols:
            flash("Choose a valid numeric target column.", "error")
        else:
            # Transparent baseline: predicts the target's mean, avoiding a misleading ML claim.
            values = df[target].dropna()
            result = {"target": target, "mean": safe_num(values.mean()), "median": safe_num(values.median()),
                      "std": safe_num(values.std()), "count": int(values.count())}
    return render_template("prediction.html", numeric_cols=numeric_cols, result=result, has_data=df is not None)

@app.route("/reports", methods=["GET", "POST"])
def reports():
    df = read_dataset()
    if request.method == "POST":
        if df is None:
            flash("Upload a dataset before generating a report.", "error")
            return redirect(url_for("reports"))
        lines = ["INSIGHTFORGE AI — DATA ANALYSIS REPORT", f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
                 f"Dataset: {session.get('dataset')}", "", "OVERVIEW",
                 json.dumps(dataset_summary(df), indent=2), "", "KEY INSIGHTS"]
        lines += [f"- {item}" for item in insights_for(df)]
        lines += ["", "COLUMN QUALITY", "Column | Type | Missing | Unique"]
        lines += [f"{x['name']} | {x['dtype']} | {x['missing']} | {x['unique']}" for x in column_quality(df)]
        output = REPORT_DIR / "insightforge_report.txt"
        output.write_text("\\n".join(lines), encoding="utf-8")
        return send_file(output, as_attachment=True, download_name="InsightForge_Analysis_Report.txt")
    return render_template("reports.html", has_data=df is not None,
                           summary=dataset_summary(df) if df is not None else None,
                           insights=insights_for(df) if df is not None else [])

@app.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("dashboard"))

@app.errorhandler(413)
def too_large(_error):
    flash("File is too large. Maximum upload size is 25 MB.", "error")
    return redirect(url_for("upload"))

if __name__ == "__main__":
    app.run(debug=True)

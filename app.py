from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "insightforge-dev-key"

# Upload configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Check allowed file extensions
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# Read CSV and Excel files
def read_dataset(file_path):
    extension = file_path.rsplit(".", 1)[1].lower()

    if extension == "csv":
        return pd.read_csv(file_path)

    return pd.read_excel(file_path)


# Analyze dataset quality
def analyze_data_quality(df):
    # Count missing values
    missing_values = int(df.isnull().sum().sum())

    # Count duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Dataset dimensions
    total_rows, total_columns = df.shape
    total_cells = total_rows * total_columns

    # Calculate data quality score
    if total_cells > 0:
        quality_score = round(
            ((total_cells - missing_values) / total_cells) * 100,
            2
        )
    else:
        quality_score = 100.0

    # Get column information
    column_info = []

    for column in df.columns:
        column_info.append({
            "name": str(column),
            "data_type": str(df[column].dtype),
            "missing": int(df[column].isnull().sum())
        })

    return {
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "quality_score": quality_score,
        "column_info": column_info
    }


# Dashboard homepage
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload-page")
def upload_page():
    return render_template("upload.html")


@app.route("/data-quality")
def data_quality_page():
    return render_template("data_quality.html")


@app.route("/data-cleaning")
def data_cleaning_page():
    return render_template("data_cleaning.html")


@app.route("/eda")
def eda_page():
    return render_template("eda.html")


@app.route("/ai-insights")
def ai_insights_page():
    return render_template("ai_insights.html")


@app.route("/ask-ai")
def ask_ai_page():
    return render_template("ask_ai.html")


@app.route("/prediction")
def prediction_page():
    return render_template("prediction.html")


@app.route("/reports")
def reports_page():
    return render_template("reports.html")


@app.route("/settings")
def settings_page():
    return render_template("settings.html")

# Clean dataset
def clean_dataset(df, missing_strategy="none", remove_duplicates=False):
    cleaned_df = df.copy()

    # Handle missing values
    if missing_strategy == "drop":
        cleaned_df = cleaned_df.dropna()

    elif missing_strategy == "zero":
        cleaned_df = cleaned_df.fillna(0)

    elif missing_strategy == "mean":
        numeric_columns = cleaned_df.select_dtypes(include="number").columns
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(
            cleaned_df[numeric_columns].mean()
        )

    elif missing_strategy == "median":
        numeric_columns = cleaned_df.select_dtypes(include="number").columns
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(
            cleaned_df[numeric_columns].median()
        )

    # Remove duplicate rows
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    return cleaned_df

# Dataset upload and analysis
@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")

    # Check whether a file was selected
    if file is None or file.filename == "":
        flash("Please select a dataset first.", "error")
        return redirect(url_for("home"))

    # Validate extension
    if not allowed_file(file.filename):
        flash("Please upload a CSV or Excel file.", "error")
        return redirect(url_for("home"))

    # Secure filename
    filename = secure_filename(file.filename)

    if not filename:
        flash("Invalid filename.", "error")
        return redirect(url_for("home"))

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:
        # Save uploaded dataset
        file.save(file_path)

        # Read dataset
        df = read_dataset(file_path)

        # Calculate dataset dimensions
        rows, columns = df.shape

        # Perform data quality analysis
        quality_report = analyze_data_quality(df)

        missing_values = quality_report["missing_values"]
        duplicate_rows = quality_report["duplicate_rows"]
        quality_score = quality_report["quality_score"]
        column_info = quality_report["column_info"]

        # Generate dataset preview
        preview_html = df.head(5).to_html(
            index=False,
            classes="data-preview",
            border=0,
            na_rep="Missing"
        )

        # Display results on dashboard
        return render_template(
            "index.html",
            dataset_name=filename,
            total_rows=rows,
            total_columns=columns,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            quality_score=quality_score,
            column_info=column_info,
            preview_table=preview_html
        )

    except Exception:
        app.logger.exception("Dataset processing failed")

        flash(
            "Could not read this dataset. Please check the file format.",
            "error"
        )

        return redirect(url_for("home"))
    # Data cleaning route
@app.route("/clean", methods=["POST"])
def clean_data():
    filename = request.form.get("dataset_name", "")

    if not filename or secure_filename(filename) != filename:
        flash("Please upload a valid dataset first.", "error")
        return redirect(url_for("home"))

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.isfile(file_path):
        flash("Dataset not found. Please upload it again.", "error")
        return redirect(url_for("home"))

    missing_strategy = request.form.get("missing_strategy", "none")
    remove_duplicates = request.form.get("remove_duplicates") == "yes"

    allowed_strategies = {"none", "drop", "zero", "mean", "median"}

    if missing_strategy not in allowed_strategies:
        flash("Invalid missing-value strategy.", "error")
        return redirect(url_for("home"))

    try:
        df = read_dataset(file_path)

        original_rows = len(df)

        cleaned_df = clean_dataset(
            df,
            missing_strategy=missing_strategy,
            remove_duplicates=remove_duplicates
        )

        cleaned_filename = f"cleaned_{filename.rsplit('.', 1)[0]}.csv"
        cleaned_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            cleaned_filename
        )

        cleaned_df.to_csv(cleaned_path, index=False)

        removed_rows = original_rows - len(cleaned_df)

        flash(
            f"Cleaning complete! {removed_rows} rows removed. "
            f"Cleaned dataset saved as {cleaned_filename}.",
            "success"
        )

        return render_template(
    "index.html",
    dataset_name=filename,
    total_rows=len(cleaned_df),
    total_columns=len(cleaned_df.columns),
    missing_values=int(cleaned_df.isnull().sum().sum()),
    duplicate_rows=int(cleaned_df.duplicated().sum()),
    quality_score=analyze_data_quality(cleaned_df)["quality_score"],
    column_info=analyze_data_quality(cleaned_df)["column_info"],
    preview_table=cleaned_df.head(5).to_html(
        index=False,
        classes="data-preview",
        border=0,
        na_rep="Missing"
    ),
    cleaned_filename=cleaned_filename
)
    except Exception:
        app.logger.exception("Dataset cleaning failed")
        flash("Could not clean the dataset. Please try again.", "error")
        return redirect(url_for("home"))
    
# Download cleaned dataset
@app.route("/download/<filename>")
def download_cleaned_file(filename):
    from flask import send_from_directory

    if not filename.startswith("cleaned_"):
        flash("Invalid download request.", "error")
        return redirect(url_for("home"))

    safe_filename = secure_filename(filename)

    if safe_filename != filename:
        flash("Invalid filename.", "error")
        return redirect(url_for("home"))

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        safe_filename
    )

    if not os.path.isfile(file_path):
        flash("Cleaned dataset not found. Please clean it again.", "error")
        return redirect(url_for("home"))

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        safe_filename,
        as_attachment=True
    )
# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)
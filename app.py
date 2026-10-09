from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "insightforge-dev-key"

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def read_dataset(file_path):
    extension = file_path.rsplit(".", 1)[1].lower()

    if extension == "csv":
        return pd.read_csv(file_path)

    return pd.read_excel(file_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")

    if file is None or file.filename == "":
        flash("Please select a dataset first.", "error")
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        flash("Please upload a CSV or Excel file.", "error")
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)

    if not filename:
        flash("Invalid filename.", "error")
        return redirect(url_for("home"))

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(file_path)
        df = read_dataset(file_path)

        rows, columns = df.shape
        missing_values = int(df.isna().sum().sum())
        total_cells = rows * columns

        if total_cells > 0:
            quality_score = round(
                ((total_cells - missing_values) / total_cells) * 100,
                2
            )
        else:
            quality_score = 100.0

        preview_html = df.head(5).to_html(
            index=False,
            classes="data-preview",
            border=0,
            na_rep="Missing"
        )

        return render_template(
            "index.html",
            dataset_name=filename,
            total_rows=rows,
            total_columns=columns,
            missing_values=missing_values,
            quality_score=quality_score,
            preview_table=preview_html
        )

    except Exception:
        app.logger.exception("Dataset processing failed")
        flash(
            "Could not read this dataset. Please check the file format.",
            "error"
        )
        return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
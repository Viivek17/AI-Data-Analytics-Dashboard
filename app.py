from flask import Flask, render_template, request
import os
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():

    file = request.files.get("file")

    if not file or file.filename == "":
        return "No file selected"

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(file_path)

    try:

        # Read CSV
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(file_path)

        # Read Excel
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)

        else:
            return "Unsupported file format"

        rows = len(df)
        columns = len(df.columns)

        preview = df.head(5).to_html(
            classes="data-preview",
            index=False
        )

        return f"""
        <h1>Dataset Uploaded Successfully</h1>

        <p><strong>File:</strong> {file.filename}</p>

        <p><strong>Total Rows:</strong> {rows}</p>

        <p><strong>Total Columns:</strong> {columns}</p>

        <h2>Dataset Preview</h2>

        {preview}

        <br>

        <a href="/">← Back to Dashboard</a>
        """

    except Exception as e:

        return f"Error reading dataset: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
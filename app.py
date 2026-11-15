import os, io, json, re
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, abort
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'local-dev-key-change-before-deployment')
BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE, 'uploads')
REPORT_FOLDER = os.path.join(BASE, 'reports')
app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, MAX_CONTENT_LENGTH=25 * 1024 * 1024)
ALLOWED = {'csv', 'xlsx', 'xls'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

def get_df():
    name = session.get('dataset_file')
    if not name:
        return None
    path = os.path.join(UPLOAD_FOLDER, secure_filename(name))
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path) if path.lower().endswith('.csv') else pd.read_excel(path)
    except Exception:
        return None

def quality(df):
    cells = max(df.shape[0] * df.shape[1], 1)
    missing, duplicates = (int(df.isna().sum().sum()), int(df.duplicated().sum()))
    score = max(0, round(100 * (1 - (missing + duplicates * max(df.shape[1], 1)) / cells)))
    return dict(rows=len(df), columns=len(df.columns), missing=missing, duplicates=duplicates, score=score, numeric=len(df.select_dtypes(include=np.number).columns), categorical=len(df.select_dtypes(include=['object', 'category', 'bool']).columns))

def render(template, active, title, **kwargs):
    return render_template(template, active_page=active, header_title=title, dataset_name=session.get('dataset_name'), **kwargs)

@app.context_processor
def context():
    return {'has_dataset': get_df() is not None}

@app.route('/')
def home():
    df = get_df()
    stats = quality(df) if df is not None else None
    preview = df.head(6).to_html(classes='data-table', index=False) if df is not None else None
    return render('index.html', 'dashboard', 'AI Data Analytics', stats=stats, preview=preview)

@app.route('/upload-page')
def upload_page():
    return render('upload.html', 'upload', 'Upload Dataset')

@app.route('/upload', methods=['POST'])
def upload_dataset():
    f = request.files.get('dataset')
    if not f or not f.filename:
        flash('Choose a dataset first.', 'error')
        return redirect(url_for('upload_page'))
    filename = secure_filename(f.filename)
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED:
        flash('Only CSV, XLSX and XLS files are supported.', 'error')
        return redirect(url_for('upload_page'))
    try:
        raw = f.read()
        df = pd.read_csv(io.BytesIO(raw)) if ext == 'csv' else pd.read_excel(io.BytesIO(raw))
        if df.empty or len(df.columns) == 0:
            raise ValueError('The dataset is empty.')
        stored = f'{datetime.now():%Y%m%d_%H%M%S}_{filename}'
        path = os.path.join(UPLOAD_FOLDER, stored)
        if ext == 'csv':
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False)
        session['dataset_file'] = stored
        session['dataset_name'] = filename
        session.pop('cleaned_file', None)
        flash(f'Dataset uploaded successfully: {len(df):,} rows × {len(df.columns)} columns.', 'success')
        return redirect(url_for('preview_page'))
    except Exception as e:
        app.logger.exception('Upload failed')
        flash(f'Unable to read dataset: {e}', 'error')
        return redirect(url_for('upload_page'))

@app.route('/preview')
def preview_page():
    df = get_df()
    if df is None:
        flash('Upload a dataset first.', 'error')
        return redirect(url_for('upload_page'))
    cols = [{'name': str(c), 'dtype': str(df[c].dtype), 'missing': int(df[c].isna().sum())} for c in df.columns]
    return render('preview.html', 'upload', 'Data Preview', df=df, stats=quality(df), columns=cols)

@app.route('/data-quality')
def data_quality_page():
    df = get_df()
    return render('data_quality.html', 'quality', 'Data Quality', df=df, stats=quality(df) if df is not None else None, missing=df.isna().sum().to_dict() if df is not None else {})

@app.route('/data-cleaning', methods=['GET', 'POST'])
def data_cleaning_page():
    df = get_df()
    if request.method == 'POST':
        if df is None:
            flash('Upload a dataset first.', 'error')
            return redirect(url_for('upload_page'))
        strategy = request.form.get('missing_strategy', 'keep')
        if strategy == 'drop_rows':
            df = df.dropna()
        elif strategy in {'mean', 'median', 'mode', 'zero'}:
            for col in df.columns:
                if not df[col].isna().any():
                    continue
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == 'mode' and (not df[col].mode().empty):
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
                elif strategy == 'zero' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
        if request.form.get('remove_duplicates') == 'yes':
            df = df.drop_duplicates()
        cleaned = 'cleaned_' + os.path.basename(session['dataset_file']).rsplit('.', 1)[0] + '.csv'
        df.to_csv(os.path.join(UPLOAD_FOLDER, cleaned), index=False)
        session['cleaned_file'] = cleaned
        flash(f'Cleaning completed. Cleaned dataset has {len(df):,} rows.', 'success')
        return redirect(url_for('data_cleaning_page'))
    cleaned = None
    name = session.get('cleaned_file')
    if name and os.path.isfile(os.path.join(UPLOAD_FOLDER, name)):
        cleaned = pd.read_csv(os.path.join(UPLOAD_FOLDER, name))
    return render('data_cleaning.html', 'cleaning', 'Data Cleaning', df=df, cleaned_df=cleaned)

@app.route('/download/<filename>')
def download_cleaned_file(filename):
    safe = secure_filename(filename)
    if safe != filename or not safe.startswith('cleaned_'):
        abort(400)
    path = os.path.join(UPLOAD_FOLDER, safe)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=safe)

@app.route('/eda', methods=['GET', 'POST'])
def eda_page():
    df = get_df()
    charts = []
    numeric = []
    categorical = []
    date_columns = []
    xcol = ycol = None
    if df is not None:
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        categorical = [c for c in df.columns if c not in numeric and df[c].nunique(dropna=True) <= 30]
        for col in df.columns:
            if col in numeric:
                continue
            parsed = pd.to_datetime(df[col], errors='coerce')
            if len(df) and parsed.notna().mean() >= 0.7:
                date_columns.append(col)

        def add_chart(title, subtitle, data, layout):
            charts.append({'title': title, 'subtitle': subtitle, 'data': data, 'layout': layout})
        for col in numeric[:2]:
            vals = pd.to_numeric(df[col], errors='coerce').dropna().tolist()
            if vals:
                add_chart(f'{col} distribution', 'Understand the spread and frequency of values.', [{'type': 'histogram', 'x': vals, 'marker': {'color': '#4778e8'}, 'nbinsx': 20}], {'xaxis': {'title': col}, 'yaxis': {'title': 'Frequency'}})
        for col in categorical[:2]:
            counts = df[col].fillna('Missing').astype(str).value_counts().head(10)
            if len(counts):
                add_chart(f'Records by {col}', 'Top categories in the uploaded dataset.', [{'type': 'bar', 'x': counts.index.tolist(), 'y': counts.values.tolist(), 'marker': {'color': '#4778e8'}}], {'xaxis': {'title': col}, 'yaxis': {'title': 'Records'}})
                if len(counts) <= 6:
                    add_chart(f'{col} composition', 'Share of the most common categories.', [{'type': 'pie', 'labels': counts.index.tolist(), 'values': counts.values.tolist(), 'hole': 0.48}], {})
        if len(numeric) >= 2:
            pair = df[numeric[:2]].apply(pd.to_numeric, errors='coerce').dropna()
            if not pair.empty:
                add_chart(f'{numeric[0]} vs {numeric[1]}', 'Explore the relationship between two numeric measures.', [{'type': 'scatter', 'mode': 'markers', 'x': pair.iloc[:, 0].tolist(), 'y': pair.iloc[:, 1].tolist(), 'marker': {'color': '#4778e8', 'size': 8, 'opacity': 0.7}}], {'xaxis': {'title': numeric[0]}, 'yaxis': {'title': numeric[1]}})
        if date_columns and numeric:
            temp = pd.DataFrame({'date': pd.to_datetime(df[date_columns[0]], errors='coerce'), 'value': pd.to_numeric(df[numeric[0]], errors='coerce')}).dropna()
            if not temp.empty:
                trend = temp.groupby(temp['date'].dt.date)['value'].mean().sort_index()
                add_chart(f'{numeric[0]} over time', f'Average {numeric[0]} grouped by {date_columns[0]}.', [{'type': 'scatter', 'mode': 'lines+markers', 'x': [str(v) for v in trend.index], 'y': trend.values.tolist(), 'line': {'color': '#4778e8', 'width': 3}}], {'xaxis': {'title': date_columns[0]}, 'yaxis': {'title': f'Average {numeric[0]}'}})
        if request.method == 'POST' and numeric:
            xcol = request.form.get('x')
            ycol = request.form.get('y')
            if xcol in df.columns and ycol in numeric:
                grouped = df.groupby(xcol, dropna=False)[ycol].mean().sort_values(ascending=False).head(15)
                add_chart(f'Average {ycol} by {xcol}', 'Custom comparison created from your selected columns.', [{'type': 'bar', 'x': [str(v) for v in grouped.index], 'y': grouped.values.tolist(), 'marker': {'color': '#20a58a'}}], {'xaxis': {'title': xcol}, 'yaxis': {'title': f'Average {ycol}'}})
        corr = df[numeric].corr().round(3).to_html(classes='data-table') if len(numeric) > 1 else None
        summary = {'rows': len(df), 'columns': len(df.columns), 'numeric': len(numeric), 'categorical': len(categorical), 'missing': int(df.isna().sum().sum())}
    else:
        corr = None
        summary = None
    return render('eda.html', 'eda', 'EDA & Visualization', df=df, numeric=numeric, categorical=categorical, correlation=corr, charts=charts, xcol=xcol, ycol=ycol, summary=summary)

@app.route('/ai-insights')
def ai_insights_page():
    df = get_df()
    insights = []
    if df is not None:
        q = quality(df)
        insights = [f'Dataset contains {len(df):,} rows and {len(df.columns)} columns.', f"There are {q['missing']:,} missing cells and {q['duplicates']:,} duplicate rows.", f"Heuristic data-quality indicator: {q['score']}%."]
        for col in df.select_dtypes(include=np.number).columns[:10]:
            s = df[col].dropna()
            if len(s):
                insights.append(f'{col}: mean {s.mean():,.4g}, median {s.median():,.4g}, minimum {s.min():,.4g}, maximum {s.max():,.4g}.')
        for col in df.select_dtypes(exclude=np.number).columns[:5]:
            counts = df[col].dropna().astype(str).value_counts().head(1)
            if len(counts):
                insights.append(f'Most frequent {col}: {counts.index[0]} ({counts.iloc[0]} records).')
    return render('ai_insights.html', 'insights', 'Dataset Insights', df=df, insights=insights)

def dataset_context(df):
    """Create a compact, privacy-conscious profile; never send the full dataset."""
    numeric = df.select_dtypes(include=np.number)
    profile = {'shape': {'rows': int(len(df)), 'columns': int(len(df.columns))}, 'columns': [{'name': str(c), 'dtype': str(df[c].dtype), 'missing': int(df[c].isna().sum()), 'unique': int(df[c].nunique(dropna=True))} for c in df.columns], 'numeric_summary': numeric.describe().replace({np.nan: None}).round(4).to_dict() if not numeric.empty else {}, 'missing_by_column': {str(k): int(v) for k, v in df.isna().sum().items()}, 'duplicate_rows': int(df.duplicated().sum()), 'top_categories': {}}
    for col in df.select_dtypes(exclude=np.number).columns[:12]:
        profile['top_categories'][str(col)] = {str(k): int(v) for k, v in df[col].astype(str).value_counts().head(8).items()}
    if len(numeric.columns) > 1:
        profile['correlations'] = numeric.corr().round(3).replace({np.nan: None}).to_dict()
    return profile

def calculate_dataset_answer(df, question):
    """Calculate common questions with Pandas before asking the LLM to explain."""
    q = question.lower()
    if any((w in q for w in ['missing', 'null', 'empty cells'])):
        vals = df.isna().sum()
        return f'Missing cells: {int(vals.sum())}. By column: ' + ', '.join((f'{k}: {int(v)}' for k, v in vals.items() if v))
    if 'duplicate' in q:
        return f'There are {int(df.duplicated().sum())} duplicate rows.'
    if any((w in q for w in ['how many rows', 'row count', 'number of rows', 'total rows'])):
        return f'The dataset has {len(df):,} rows.'
    if any((w in q for w in ['how many columns', 'column count', 'number of columns', 'total columns'])):
        return f"The dataset has {len(df.columns)} columns: {', '.join(map(str, df.columns))}."
    if 'column' in q and ('list' in q or 'what are' in q or 'names' in q):
        return 'Columns: ' + ', '.join(map(str, df.columns))
    if 'correlation' in q or 'correlate' in q:
        n = df.select_dtypes(include=np.number)
        if n.shape[1] < 2:
            return 'At least two numeric columns are required for correlation analysis.'
        corr = n.corr()
        pairs = []
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                if pd.notna(corr.loc[a, b]):
                    pairs.append((abs(float(corr.loc[a, b])), a, b, float(corr.loc[a, b])))
        if not pairs:
            return 'No valid numeric correlations could be calculated.'
        pairs.sort(reverse=True)
        return 'Strongest numeric relationships: ' + '; '.join((f'{a} vs {b}: {v:.3f}' for _, a, b, v in pairs[:5]))
    col = next((c for c in sorted(df.columns, key=lambda x: len(str(x)), reverse=True) if str(c).lower() in q), None)
    if col is None:
        return None
    if pd.api.types.is_numeric_dtype(df[col]):
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if values.empty:
            return f'Column {col} has no usable numeric values.'
        if any((w in q for w in ['average', 'mean', 'avg'])):
            return f'Average {col}: {values.mean():,.4f} (based on {len(values):,} non-missing values).'
        if any((w in q for w in ['median'])):
            return f'Median {col}: {values.median():,.4f}.'
        if any((w in q for w in ['highest', 'maximum', 'max', 'largest', 'top'])):
            idx = values.idxmax()
            return f'Highest {col}: {values.loc[idx]:,.4f} (row index {idx}).'
        if any((w in q for w in ['lowest', 'minimum', 'min', 'smallest'])):
            idx = values.idxmin()
            return f'Lowest {col}: {values.loc[idx]:,.4f} (row index {idx}).'
        return f'Summary for {col}: count {len(values):,}, mean {values.mean():,.4f}, median {values.median():,.4f}, min {values.min():,.4f}, max {values.max():,.4f}, standard deviation {values.std():,.4f}.'
    counts = df[col].fillna('Missing').astype(str).value_counts().head(10)
    if any((w in q for w in ['highest', 'most', 'top', 'best', 'popular', 'frequent'])):
        return f'Top values in {col}: ' + ', '.join((f'{k} ({v:,})' for k, v in counts.items()))
    return f'Column {col} has {df[col].nunique(dropna=True):,} distinct non-missing values. Most common: ' + ', '.join((f'{k} ({v:,})' for k, v in counts.head(5).items()))

def generate_chat_reply(df, question, history):
    calculated = calculate_dataset_answer(df, question)
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            profile = dataset_context(df)
            messages = [{'role': 'system', 'content': 'You are InsightForge AI, a helpful data-analysis assistant. Answer in clear, beginner-friendly language. Use ONLY the supplied dataset profile and calculated result as evidence. Never invent values or claim to have inspected raw rows. If the profile/calculation does not answer the question, say what is missing and suggest a supported question. Explain statistics briefly and mention limitations where relevant. Do not generate or execute code.\nDATASET_PROFILE=' + json.dumps(profile, ensure_ascii=False) + '\nCALCULATED_RESULT=' + (calculated or 'No deterministic calculation matched this question.')}]
            for item in history[-4:]:
                if item.get('role') in ('user', 'assistant') and isinstance(item.get('content'), str):
                    messages.append({'role': item['role'], 'content': item['content'][:2000]})
            messages.append({'role': 'user', 'content': question[:2000]})
            response = client.chat.completions.create(model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'), messages=messages, temperature=0.2, max_tokens=600)
            return (response.choices[0].message.content.strip(), 'AI assistant')
        except Exception as exc:
            app.logger.warning('LLM request failed: %s', exc)
            if calculated:
                return (calculated + '\n\n(LLM service is currently unavailable, so this answer is based on a direct Pandas calculation.)', 'Pandas analysis')
            return ("I couldn't connect to the AI service. Check your API key, internet connection, and model setting. You can still ask direct questions about column statistics, missing values, duplicates, and correlations.", 'Offline assistant')
    if calculated:
        return (calculated + '\n\nTip: add OPENAI_API_KEY to your .env file to enable conversational AI explanations. The current answer was calculated directly from your dataset.', 'Pandas analysis')
    return ('I can help you explore this dataset. Try asking: ‘What is the average Sales?’, ‘Which region has the most records?’, ‘Show missing values’, ‘Which columns are correlated?’, or ‘How many rows are there?’\n\nFor open-ended conversational answers, configure OPENAI_API_KEY in your .env file.', 'Offline assistant')

@app.route('/ask-ai', methods=['GET'])
def ask_ai_page():
    df = get_df()
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render('ask_ai.html', 'ask_ai', 'Ask Your Data', df=df, chat_history=session.get('chat_history', []), ai_enabled=bool(os.getenv('OPENAI_API_KEY', '').strip()))

@app.route('/ask-ai/chat', methods=['POST'])
def ask_ai_chat():
    df = get_df()
    if df is None:
        return ({'error': 'Upload a dataset first.'}, 400)
    payload = request.get_json(silent=True) or {}
    question = str(payload.get('message', '')).strip()
    if not question:
        return ({'error': 'Please enter a question.'}, 400)
    if len(question) > 2000:
        return ({'error': 'Please keep your question under 2,000 characters.'}, 400)
    history = session.get('chat_history', [])
    answer, source = generate_chat_reply(df, question, history)
    history.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer, 'source': source}])
    session['chat_history'] = history[-4:]
    session.modified = True
    return {'answer': answer, 'source': source}

@app.route('/ask-ai/clear', methods=['POST'])
def clear_chat():
    session.pop('chat_history', None)
    return {'ok': True}

@app.route('/prediction', methods=['GET', 'POST'])
def prediction_page():
    df = get_df()
    result = None
    if request.method == 'POST':
        try:
            if df is None:
                raise ValueError('Upload a dataset first.')
            target = request.form.get('target', '')
            task = request.form.get('task', 'regression')
            algorithm = request.form.get('algorithm', 'random_forest')
            if target not in df.columns:
                raise ValueError('Choose a valid target column.')
            if task not in {'regression', 'classification'}:
                raise ValueError('Choose a valid prediction type.')
            data = df.copy()
            y = data[target]
            if task == 'regression':
                if not pd.api.types.is_numeric_dtype(y):
                    cleaned_target = y.astype(str).str.replace(',', '', regex=False)
                    extracted = cleaned_target.str.extract(r'([-+]?(?:\d+\.?\d*|\.\d+))', expand=False)
                    y = pd.to_numeric(extracted, errors='coerce')
                else:
                    y = pd.to_numeric(y, errors='coerce')
                valid = y.notna() & np.isfinite(y)
                data, y = (data.loc[valid].copy(), y.loc[valid])
                if len(data) < 4:
                    raise ValueError(f'Regression needs at least 4 valid numeric target values; found {len(data)}. Check the selected target column or upload more rows.')
                if y.nunique() < 2:
                    raise ValueError('The selected regression target has only one unique value. Choose a target with variation.')
            else:
                data = data.dropna(subset=[target]).copy()
                y = data[target]
                if len(data) < 4:
                    raise ValueError(f'Classification needs at least 4 non-missing target rows; found {len(data)}.')
                y = y.astype(str)
                counts = y.value_counts()
                if len(counts) < 2 or counts.min() < 2:
                    raise ValueError('Classification needs at least two classes with two rows in each class.')
            X = data.drop(columns=[target]).copy()
            for col in list(X.columns):
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    dt = pd.to_datetime(X[col], errors='coerce')
                    X[col + '_year'] = dt.dt.year
                    X[col + '_month'] = dt.dt.month
                    X[col + '_day'] = dt.dt.day
                    X.drop(columns=[col], inplace=True)
                elif X[col].dtype == 'object':
                    parsed = pd.to_datetime(X[col], errors='coerce')
                    if parsed.notna().mean() >= 0.8:
                        X[col + '_year'] = parsed.dt.year
                        X[col + '_month'] = parsed.dt.month
                        X[col + '_day'] = parsed.dt.day
                        X.drop(columns=[col], inplace=True)
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=True) <= 1])
            if X.empty:
                raise ValueError('No useful feature columns remain after removing constant columns.')
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = [c for c in X.columns if c not in numeric_cols]
            for col in categorical_cols:
                X[col] = X[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
            transformers = []
            if numeric_cols:
                transformers.append(('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric_cols))
            if categorical_cols:
                transformers.append(('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols))
            if not transformers:
                raise ValueError('No usable numeric or categorical features were found.')
            prep = ColumnTransformer(transformers)
            model_options = {'random_forest': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1) if task == 'regression' else RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1), 'extra_trees': ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1) if task == 'regression' else ExtraTreesClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1), 'decision_tree': DecisionTreeRegressor(max_depth=6, random_state=42) if task == 'regression' else DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced'), 'linear': LinearRegression() if task == 'regression' else LogisticRegression(max_iter=1000, class_weight='balanced')}
            if algorithm not in model_options:
                raise ValueError('Choose a valid machine-learning algorithm.')
            model = model_options[algorithm]
            stratify = y if task == 'classification' else None
            test_size = max(1, int(np.ceil(len(y) * 0.2)))
            if task == 'classification':
                counts = y.value_counts()
                if len(counts) < 2 or counts.min() < 2:
                    raise ValueError('Classification needs at least two classes, with at least 2 examples in each class.')
                test_count = max(int(np.ceil(len(y) * 0.2)), y.nunique())
                if len(y) - test_count < y.nunique():
                    raise ValueError('There are not enough examples per class for a train/test split. Add more rows for each class.')
                test_size = test_count
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
            pipeline = Pipeline([('preprocessor', prep), ('model', model)])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            charts = []
            if task == 'regression':
                mae = mean_absolute_error(y_test, predictions)
                rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
                r2 = r2_score(y_test, predictions) if len(y_test) > 1 else float('nan')
                metrics = {'MAE': round(float(mae), 4), 'RMSE': round(rmse, 4), 'R²': round(float(r2), 4)}
                compare_n = min(25, len(y_test))
                compare = go.Figure()
                compare.add_trace(go.Bar(name='Actual', x=list(range(1, compare_n + 1)), y=np.asarray(y_test)[:compare_n].tolist()))
                compare.add_trace(go.Bar(name='Predicted', x=list(range(1, compare_n + 1)), y=np.asarray(predictions)[:compare_n].tolist()))
                compare.update_layout(title='Actual vs Predicted (test samples)', barmode='group', xaxis_title='Test sample', yaxis_title=target, template='plotly_white', height=390, margin=dict(l=20, r=20, t=55, b=35))
                charts.append({'title': 'Prediction comparison', 'json': compare.to_json()})
                residuals = np.asarray(y_test, dtype=float) - np.asarray(predictions, dtype=float)
                residual_fig = go.Figure(go.Histogram(x=residuals, nbinsx=20, marker_color='#4f7cff'))
                residual_fig.update_layout(title='Prediction error distribution', xaxis_title='Actual − predicted', yaxis_title='Number of rows', template='plotly_white', height=330, margin=dict(l=20, r=20, t=55, b=35))
                charts.append({'title': 'Prediction errors', 'json': residual_fig.to_json()})
                note = f'{type(model).__name__} evaluated on {len(X_test)} unseen test rows. Lower MAE/RMSE are generally better; R² can be negative and should be interpreted in context.'
                if len(data) < 30:
                    note += ' Warning: this is a small dataset, so metrics may vary substantially with a different split. Treat results as a demonstration, not a reliable production estimate.'
            else:
                metrics = {'Accuracy': round(float(accuracy_score(y_test, predictions)), 4), 'Precision (weighted)': round(float(precision_score(y_test, predictions, average='weighted', zero_division=0)), 4), 'Recall (weighted)': round(float(recall_score(y_test, predictions, average='weighted', zero_division=0)), 4), 'F1 (weighted)': round(float(f1_score(y_test, predictions, average='weighted', zero_division=0)), 4)}
                labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
                actual_counts = pd.Series(y_test).astype(str).value_counts()
                predicted_counts = pd.Series(predictions).astype(str).value_counts()
                class_fig = go.Figure()
                class_fig.add_trace(go.Bar(name='Actual', x=labels, y=[int(actual_counts.get(label, 0)) for label in labels]))
                class_fig.add_trace(go.Bar(name='Predicted', x=labels, y=[int(predicted_counts.get(label, 0)) for label in labels]))
                class_fig.update_layout(title='Actual vs Predicted class counts', barmode='group', xaxis_title='Class', yaxis_title='Rows', template='plotly_white', height=380, margin=dict(l=20, r=20, t=55, b=35))
                charts.append({'title': 'Class comparison', 'json': class_fig.to_json()})
                cm = confusion_matrix(y_test, predictions, labels=labels)
                cm_fig = go.Figure(go.Heatmap(z=cm.tolist(), x=labels, y=labels, colorscale='Blues', text=cm.tolist(), texttemplate='%{text}'))
                cm_fig.update_layout(title='Confusion matrix', xaxis_title='Predicted label', yaxis_title='Actual label', template='plotly_white', height=380, margin=dict(l=20, r=20, t=55, b=35))
                charts.append({'title': 'Confusion matrix', 'json': cm_fig.to_json()})
                note = f'{type(model).__name__} evaluated on {len(X_test)} unseen test rows. Weighted metrics account for class sizes; check the confusion matrix for class-specific errors.'
                if len(data) < 30:
                    note += ' Warning: this is a small dataset, so metrics may vary substantially with a different split. Treat results as a demonstration, not a reliable production estimate.'
            estimator = pipeline.named_steps['model']
            if hasattr(estimator, 'feature_importances_'):
                try:
                    names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                    importances = estimator.feature_importances_
                    top_idx = np.argsort(importances)[-12:]
                    imp_fig = go.Figure(go.Bar(x=importances[top_idx].tolist(), y=[str(names[i]).replace('numeric__', '').replace('categorical__', '') for i in top_idx], orientation='h', marker_color='#4f7cff'))
                    imp_fig.update_layout(title='Most influential model features', xaxis_title='Feature importance', yaxis_title='Feature', template='plotly_white', height=380, margin=dict(l=20, r=20, t=55, b=35))
                    charts.append({'title': 'Feature importance', 'json': imp_fig.to_json()})
                except Exception:
                    pass
            result = {'metrics': metrics, 'note': note, 'charts': charts, 'algorithm': type(model).__name__, 'target': target, 'task': task}
        except Exception as e:
            app.logger.exception('Prediction training failed')
            result = {'error': str(e)}
    return render('prediction.html', 'prediction', 'Machine Learning', df=df, columns=df.columns.tolist() if df is not None else [], result=result)

@app.route('/reports', methods=['GET', 'POST'])
def reports_page():
    df = get_df()
    generated = False
    if request.method == 'POST':
        if df is None:
            flash('Upload a dataset first.', 'error')
        else:
            path = os.path.join(REPORT_FOLDER, 'InsightForge_Report.pdf')
            c = canvas.Canvas(path, pagesize=A4)
            _, h = A4
            y = h - 55
            c.setFont('Helvetica-Bold', 18)
            c.drawString(45, y, 'InsightForge AI - Analysis Report')
            y -= 35
            c.setFont('Helvetica', 10)
            lines = [f"Dataset: {session.get('dataset_name', 'Uploaded dataset')}", f'Generated: {datetime.now():%Y-%m-%d %H:%M}', f'Rows: {len(df)} | Columns: {len(df.columns)}', f'Missing cells: {int(df.isna().sum().sum())}', f'Duplicate rows: {int(df.duplicated().sum())}', '', 'Descriptive statistics:'] + df.describe().round(3).to_string().splitlines()[:24]
            for line in lines:
                if y < 55:
                    c.showPage()
                    y = h - 55
                    c.setFont('Helvetica', 9)
                c.drawString(45, y, str(line)[:105])
                y -= 15
            c.save()
            generated = True
            flash('PDF report generated.', 'success')
    return render('reports.html', 'reports', 'Reports', df=df, generated=generated)

@app.route('/reports/download')
def download_report():
    p = os.path.join(REPORT_FOLDER, 'InsightForge_Report.pdf')
    if not os.path.isfile(p):
        abort(404)
    return send_file(p, as_attachment=True, download_name='InsightForge_Report.pdf')

@app.route('/settings', methods=['GET', 'POST'])
def settings_page():
    if request.method == 'POST':
        session.clear()
        flash('Active dataset session cleared. Uploaded files remain in the local uploads folder.', 'success')
        return redirect(url_for('settings_page'))
    return render('settings.html', 'settings', 'Settings')

@app.errorhandler(413)
def too_large(_):
    flash('File exceeds the 25 MB upload limit.', 'error')
    return redirect(url_for('upload_page'))
if __name__ == '__main__':
    app.run(debug=True)

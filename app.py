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
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Image as RLImage
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.utils import ImageReader
from io import BytesIO
import html
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'local-dev-key-change-before-deployment')
BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE, 'uploads')
REPORT_FOLDER = os.path.join(BASE, 'reports')
app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, MAX_CONTENT_LENGTH=25 * 1024 * 1024)
ALLOWED = {'csv', 'xlsx', 'xls'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True); os.makedirs(REPORT_FOLDER, exist_ok=True)

def get_df():
    name = session.get('dataset_file')
    if not name: return None
    path = os.path.join(UPLOAD_FOLDER, secure_filename(name))
    if not os.path.isfile(path): return None
    try: return pd.read_csv(path) if path.lower().endswith('.csv') else pd.read_excel(path)
    except Exception: return None

def quality(df):
    cells = max(df.shape[0] * df.shape[1], 1)
    missing, duplicates = int(df.isna().sum().sum()), int(df.duplicated().sum())
    score = max(0, round(100 * (1 - (missing + duplicates * max(df.shape[1], 1)) / cells)))
    return dict(rows=len(df), columns=len(df.columns), missing=missing, duplicates=duplicates,
                score=score, numeric=len(df.select_dtypes(include=np.number).columns),
                categorical=len(df.select_dtypes(include=['object','category','bool']).columns))

def render(template, active, title, **kwargs):
    return render_template(template, active_page=active, header_title=title,
                           dataset_name=session.get('dataset_name'), **kwargs)

@app.context_processor
def context(): return {'has_dataset': get_df() is not None}

@app.route('/')
def home():
    df=get_df(); stats=quality(df) if df is not None else None
    preview=df.head(6).to_html(classes='data-table', index=False) if df is not None else None
    return render('index.html','dashboard','AI Data Analytics',stats=stats,preview=preview)

@app.route('/upload-page')
def upload_page(): return render('upload.html','upload','Upload Dataset')

@app.route('/upload', methods=['POST'])
def upload_dataset():
    f=request.files.get('dataset')
    if not f or not f.filename:
        flash('Choose a dataset first.','error'); return redirect(url_for('upload_page'))
    filename=secure_filename(f.filename); ext=filename.rsplit('.',1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED:
        flash('Only CSV, XLSX and XLS files are supported.','error'); return redirect(url_for('upload_page'))
    try:
        raw=f.read(); df=pd.read_csv(io.BytesIO(raw)) if ext=='csv' else pd.read_excel(io.BytesIO(raw))
        if df.empty or len(df.columns)==0: raise ValueError('The dataset is empty.')
        stored=f"{datetime.now():%Y%m%d_%H%M%S}_{filename}"
        path=os.path.join(UPLOAD_FOLDER,stored)
        if ext=='csv': df.to_csv(path,index=False)
        else: df.to_excel(path,index=False)
        session['dataset_file']=stored; session['dataset_name']=filename; session.pop('cleaned_file',None)
        flash(f'Dataset uploaded successfully: {len(df):,} rows × {len(df.columns)} columns.','success')
        return redirect(url_for('preview_page'))
    except Exception as e:
        app.logger.exception('Upload failed'); flash(f'Unable to read dataset: {e}','error')
        return redirect(url_for('upload_page'))

@app.route('/preview')
def preview_page():
    df=get_df()
    if df is None: flash('Upload a dataset first.','error'); return redirect(url_for('upload_page'))
    cols=[{'name':str(c),'dtype':str(df[c].dtype),'missing':int(df[c].isna().sum())} for c in df.columns]
    return render('preview.html','upload','Data Preview',df=df,stats=quality(df),columns=cols)

@app.route('/data-quality')
def data_quality_page():
    df=get_df(); return render('data_quality.html','quality','Data Quality',df=df,
        stats=quality(df) if df is not None else None, missing=df.isna().sum().to_dict() if df is not None else {})

@app.route('/data-cleaning', methods=['GET', 'POST'])
def data_cleaning_page():
    original = get_df()
    cleaned = None
    cleaning_summary = session.get('cleaning_summary')
    if original is None:
        if request.method == 'POST':
            flash('Upload a dataset before cleaning it.', 'error')
            return redirect(url_for('upload_page'))
        return render('data_cleaning.html', 'cleaning', 'Data Cleaning', df=None,
                      cleaned_df=None, cleaning_summary=None, column_profile=[])

    if request.method == 'POST':
        df = original.copy(deep=True)
        before_rows, before_cols = df.shape
        before_missing = int(df.isna().sum().sum())
        before_duplicates = int(df.duplicated().sum())
        operations = []

        # Normalize text values without changing the uploaded source file.
        if request.form.get('trim_text') == 'yes':
            text_cols = df.select_dtypes(include=['object', 'string']).columns
            for col in text_cols:
                df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)
            operations.append(f'Trimmed leading/trailing spaces in {len(text_cols)} text column(s)')

        if request.form.get('standardize_missing') == 'yes':
            tokens = {'', 'na', 'n/a', 'null', 'none', 'nan', '-', '--'}
            for col in df.select_dtypes(include=['object', 'string']).columns:
                df[col] = df[col].map(lambda value: np.nan if isinstance(value, str) and value.strip().lower() in tokens else value)
            operations.append('Standardized common missing-value markers')

        # Convert selected columns only when conversion is requested.
        for col in request.form.getlist('convert_numeric'):
            if col in df.columns:
                converted = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9eE+\-.]', '', regex=True), errors='coerce')
                if converted.notna().sum() > 0:
                    df[col] = converted
                    operations.append(f'Converted {col} to numeric')
        for col in request.form.getlist('convert_date'):
            if col in df.columns:
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notna().sum() > 0:
                    df[col] = converted
                    operations.append(f'Converted {col} to date')

        strategy = request.form.get('missing_strategy', 'keep')
        if strategy == 'drop_rows':
            old_len = len(df); df = df.dropna(); operations.append(f'Removed {old_len - len(df):,} rows containing missing values')
        elif strategy in {'mean', 'median', 'mode', 'zero', 'ffill', 'bfill', 'constant'}:
            for col in df.columns:
                if not df[col].isna().any():
                    continue
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].mean()
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].median()
                elif strategy == 'mode':
                    modes = df[col].mode(dropna=True); fill_value = modes.iloc[0] if not modes.empty else None
                elif strategy == 'zero' and pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = 0
                elif strategy == 'constant':
                    fill_value = request.form.get('constant_value', 'Unknown')
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = pd.to_numeric(fill_value, errors='coerce')
                else:
                    fill_value = None
                if strategy == 'ffill': df[col] = df[col].ffill()
                elif strategy == 'bfill': df[col] = df[col].bfill()
                elif fill_value is not None and not (isinstance(fill_value, float) and np.isnan(fill_value)):
                    df[col] = df[col].fillna(fill_value)
            operations.append(f'Applied {strategy} missing-value strategy where compatible')

        if request.form.get('remove_duplicates') == 'yes':
            old_len = len(df); df = df.drop_duplicates(); operations.append(f'Removed {old_len - len(df):,} duplicate rows')

        if request.form.get('drop_empty_columns') == 'yes':
            empty_cols = df.columns[df.isna().all()].tolist()
            if empty_cols:
                df = df.drop(columns=empty_cols); operations.append(f'Removed {len(empty_cols)} fully empty column(s)')

        outlier_method = request.form.get('outlier_method', 'none')
        outlier_cols = [c for c in request.form.getlist('outlier_columns') if c in df.select_dtypes(include=np.number).columns]
        if outlier_method in {'remove', 'cap'}:
            changed = 0
            for col in outlier_cols:
                series = pd.to_numeric(df[col], errors='coerce')
                q1, q3 = series.quantile(.25), series.quantile(.75)
                if pd.isna(q1) or pd.isna(q3): continue
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = series.notna() & ((series < lower) | (series > upper))
                changed += int(mask.sum())
                if outlier_method == 'cap': df[col] = series.clip(lower, upper)
                else: df = df.loc[~mask].copy()
            operations.append(('Capped' if outlier_method == 'cap' else 'Removed') + f' {changed:,} IQR outlier value(s) across selected columns')

        # Keep a safe, unique output name; never overwrite the uploaded source.
        base_name = os.path.splitext(os.path.basename(session.get('dataset_file', 'dataset')))[0]
        cleaned_name = f"cleaned_{base_name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        output_path = os.path.join(UPLOAD_FOLDER, cleaned_name)
        df.to_csv(output_path, index=False)
        session['cleaned_file'] = cleaned_name
        summary = {
            'before_rows': int(before_rows), 'after_rows': int(len(df)),
            'before_cols': int(before_cols), 'after_cols': int(len(df.columns)),
            'before_missing': before_missing, 'after_missing': int(df.isna().sum().sum()),
            'before_duplicates': before_duplicates, 'after_duplicates': int(df.duplicated().sum()),
            'operations': operations, 'output_name': cleaned_name,
        }
        session['cleaning_summary'] = summary
        flash('Cleaning completed. The original uploaded dataset has not been changed.', 'success')
        return redirect(url_for('data_cleaning_page'))

    name = session.get('cleaned_file')
    if name and os.path.isfile(os.path.join(UPLOAD_FOLDER, secure_filename(name))):
        try: cleaned = pd.read_csv(os.path.join(UPLOAD_FOLDER, secure_filename(name)))
        except Exception: cleaned = None
    column_profile = [{'name': str(c), 'dtype': str(original[c].dtype),
                       'missing': int(original[c].isna().sum()),
                       'unique': int(original[c].nunique(dropna=True)),
                       'numeric': bool(pd.api.types.is_numeric_dtype(original[c]))}
                      for c in original.columns]
    return render('data_cleaning.html', 'cleaning', 'Data Cleaning', df=original,
                  cleaned_df=cleaned, cleaning_summary=cleaning_summary,
                  column_profile=column_profile)

@app.route('/download/<filename>')
def download_cleaned_file(filename):
    safe=secure_filename(filename)
    if safe!=filename or not safe.startswith('cleaned_'): abort(400)
    path=os.path.join(UPLOAD_FOLDER,safe)
    if not os.path.isfile(path): abort(404)
    return send_file(path,as_attachment=True,download_name=safe)

@app.route('/eda',methods=['GET','POST'])
def eda_page():
    df=get_df(); charts=[]; numeric=[]; categorical=[]; date_columns=[]; xcol=ycol=None
    if df is not None:
        numeric=df.select_dtypes(include=np.number).columns.tolist()
        categorical=[c for c in df.columns if c not in numeric and df[c].nunique(dropna=True)<=30]
        # Detect date-like columns without changing the uploaded dataframe.
        for col in df.columns:
            if col in numeric: continue
            parsed=pd.to_datetime(df[col],errors='coerce')
            if len(df) and parsed.notna().mean()>=0.7: date_columns.append(col)

        def add_chart(title, subtitle, data, layout):
            charts.append({'title':title,'subtitle':subtitle,'data':data,'layout':layout})

        # Numeric distributions
        for col in numeric[:2]:
            vals=pd.to_numeric(df[col],errors='coerce').dropna().tolist()
            if vals:
                add_chart(f'{col} distribution','Understand the spread and frequency of values.',
                    [{'type':'histogram','x':vals,'marker':{'color':'#4778e8'},'nbinsx':20}],
                    {'xaxis':{'title':col},'yaxis':{'title':'Frequency'}})

        # Category comparisons and composition
        for col in categorical[:2]:
            counts=df[col].fillna('Missing').astype(str).value_counts().head(10)
            if len(counts):
                add_chart(f'Records by {col}','Top categories in the uploaded dataset.',
                    [{'type':'bar','x':counts.index.tolist(),'y':counts.values.tolist(),'marker':{'color':'#4778e8'}}],
                    {'xaxis':{'title':col},'yaxis':{'title':'Records'}})
                if len(counts)<=6:
                    add_chart(f'{col} composition','Share of the most common categories.',
                        [{'type':'pie','labels':counts.index.tolist(),'values':counts.values.tolist(),'hole':0.48}],{})

        # Relationship between the first two numeric columns
        if len(numeric)>=2:
            pair=df[numeric[:2]].apply(pd.to_numeric,errors='coerce').dropna()
            if not pair.empty:
                add_chart(f'{numeric[0]} vs {numeric[1]}','Explore the relationship between two numeric measures.',
                    [{'type':'scatter','mode':'markers','x':pair.iloc[:,0].tolist(),'y':pair.iloc[:,1].tolist(),
                      'marker':{'color':'#4778e8','size':8,'opacity':0.7}}],
                    {'xaxis':{'title':numeric[0]},'yaxis':{'title':numeric[1]}})

        # Time trend: average of first numeric measure by date
        if date_columns and numeric:
            temp=pd.DataFrame({'date':pd.to_datetime(df[date_columns[0]],errors='coerce'),
                               'value':pd.to_numeric(df[numeric[0]],errors='coerce')}).dropna()
            if not temp.empty:
                trend=temp.groupby(temp['date'].dt.date)['value'].mean().sort_index()
                add_chart(f'{numeric[0]} over time',f'Average {numeric[0]} grouped by {date_columns[0]}.',
                    [{'type':'scatter','mode':'lines+markers','x':[str(v) for v in trend.index],
                      'y':trend.values.tolist(),'line':{'color':'#4778e8','width':3}}],
                    {'xaxis':{'title':date_columns[0]},'yaxis':{'title':f'Average {numeric[0]}'}})

        if request.method=='POST' and numeric:
            xcol=request.form.get('x'); ycol=request.form.get('y')
            if xcol in df.columns and ycol in numeric:
                grouped=df.groupby(xcol,dropna=False)[ycol].mean().sort_values(ascending=False).head(15)
                add_chart(f'Average {ycol} by {xcol}','Custom comparison created from your selected columns.',
                    [{'type':'bar','x':[str(v) for v in grouped.index],'y':grouped.values.tolist(),
                      'marker':{'color':'#20a58a'}}],
                    {'xaxis':{'title':xcol},'yaxis':{'title':f'Average {ycol}'}})

        corr=df[numeric].corr().round(3).to_html(classes='data-table') if len(numeric)>1 else None
        summary={'rows':len(df),'columns':len(df.columns),'numeric':len(numeric),
                 'categorical':len(categorical),'missing':int(df.isna().sum().sum())}
    else:
        corr=None; summary=None
    return render('eda.html','eda','EDA & Visualization',df=df,numeric=numeric,
                  categorical=categorical,correlation=corr,charts=charts,xcol=xcol,ycol=ycol,summary=summary)

@app.route('/ai-insights')
def ai_insights_page():
    df = get_df()
    findings, charts, metrics, recommendations = [], [], [], []
    profile = None
    if df is not None:
        q = quality(df)
        numeric = df.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan)
        categorical = [c for c in df.columns if c not in numeric.columns and df[c].nunique(dropna=True) <= 50]
        total_cells = max(int(df.shape[0] * df.shape[1]), 1)
        missing_pct = round(q['missing'] / total_cells * 100, 1)
        metrics = [
            {'label':'Records analysed','value':f"{len(df):,}",'note':'Rows in the uploaded dataset'},
            {'label':'Data fields','value':str(len(df.columns)),'note':f"{len(numeric.columns)} numeric · {len(categorical)} categorical"},
            {'label':'Missing values','value':f"{q['missing']:,}",'note':f'{missing_pct}% of all cells'},
            {'label':'Data quality indicator','value':f"{q['score']}%",'note':'Heuristic completeness / duplication score'}
        ]
        profile = {'name': session.get('dataset_name') or 'Uploaded dataset', 'rows':len(df), 'columns':len(df.columns), 'duplicates':q['duplicates']}

        if q['missing']:
            worst_col = df.isna().sum().idxmax()
            worst_count = int(df[worst_col].isna().sum())
            findings.append({'type':'warning','tag':'DATA QUALITY','title':'Missing values detected',
                'text':f"{q['missing']:,} cells ({missing_pct}% of the dataset) are empty. '{worst_col}' has the highest count with {worst_count:,} missing values."})
            recommendations.append(f"Review missing values in '{worst_col}' first; choose an imputation strategy based on what the field represents.")
        else:
            findings.append({'type':'positive','tag':'DATA QUALITY','title':'No missing cells detected','text':'Every cell in the uploaded dataset contains a value.'})
        if q['duplicates']:
            findings.append({'type':'warning','tag':'DATA QUALITY','title':'Duplicate records found','text':f"{q['duplicates']:,} rows are exact duplicates. Confirm whether they are repeated events before removing them."})
            recommendations.append('Inspect duplicate rows and remove them only if they are unintended repeats.')
        else:
            findings.append({'type':'positive','tag':'DATA QUALITY','title':'No duplicate rows detected','text':'The dataset contains no exact duplicate rows.'})

        # Missing-value chart
        miss = df.isna().sum().sort_values(ascending=True)
        miss = miss[miss > 0].tail(12)
        if not miss.empty:
            fig = go.Figure(go.Bar(x=miss.values.tolist(), y=[str(x) for x in miss.index], orientation='h', marker_color='#e5a044'))
            fig.update_layout(title='Columns with missing values', xaxis_title='Missing cells', yaxis_title='', template='plotly_white', height=330, margin=dict(l=20,r=20,t=50,b=35))
            charts.append({'title':'Missing values by field','json':fig.to_json()})

        # Numeric summaries and distribution insights
        numeric_stats = []
        for col in numeric.columns[:20]:
            values = numeric[col].dropna()
            if values.empty: continue
            mean, median = float(values.mean()), float(values.median())
            std = float(values.std()) if len(values) > 1 else 0.0
            numeric_stats.append({'field':str(col),'count':int(values.size),'mean':mean,'median':median,'min':float(values.min()),'max':float(values.max()),'std':std})
            skew = float(values.skew()) if len(values) > 2 else 0.0
            if np.isfinite(skew) and abs(skew) >= 1:
                direction = 'right' if skew > 0 else 'left'
                findings.append({'type':'info','tag':'DISTRIBUTION','title':f'{col} is {direction}-skewed',
                    'text':f"The distribution is asymmetric (skewness {skew:.2f}). The mean ({mean:,.3g}) and median ({median:,.3g}) may describe the typical value differently."})
            if std > 0 and len(values) >= 5:
                z = (values - mean) / std
                unusual = int((z.abs() > 3).sum())
                if unusual:
                    findings.append({'type':'warning','tag':'POTENTIAL ANOMALY','title':f'Unusual values in {col}',
                        'text':f"{unusual:,} values are more than three standard deviations from the mean. These are candidates for review, not automatically errors."})
            if len(numeric_stats) <= 8:
                pass
        if numeric_stats:
            summary_rows = numeric_stats[:10]
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Mean', x=[x['field'] for x in summary_rows], y=[x['mean'] for x in summary_rows], marker_color='#3268e8'))
            fig.add_trace(go.Bar(name='Median', x=[x['field'] for x in summary_rows], y=[x['median'] for x in summary_rows], marker_color='#20a58a'))
            fig.update_layout(title='Mean vs median by numeric field', barmode='group', xaxis_title='Numeric field', yaxis_title='Value', template='plotly_white', height=360, margin=dict(l=20,r=20,t=50,b=75), legend=dict(orientation='h',y=1.12))
            charts.append({'title':'Numeric field comparison','json':fig.to_json()})
            for item in numeric_stats[:8]:
                findings.append({'type':'info','tag':'NUMERIC SUMMARY','title':item['field'],
                    'text':f"{item['count']:,} valid values · mean {item['mean']:,.4g} · median {item['median']:,.4g} · range {item['min']:,.4g} to {item['max']:,.4g}."})

        # Category leaders
        category_summary = []
        for col in categorical[:15]:
            counts = df[col].dropna().astype(str).value_counts()
            if counts.empty: continue
            top_label, top_count = str(counts.index[0]), int(counts.iloc[0])
            share = top_count / max(int(counts.sum()), 1) * 100
            category_summary.append((str(col), top_label, top_count, share, counts))
            findings.append({'type':'info','tag':'CATEGORY MIX','title':f"Leading value in {col}",'text':f"'{top_label}' appears {top_count:,} times ({share:.1f}% of non-empty values)."})
        if category_summary:
            col, _, _, _, counts = category_summary[0]
            top = counts.head(8).sort_values()
            fig = go.Figure(go.Bar(x=top.values.tolist(), y=[str(x) for x in top.index], orientation='h', marker_color='#3268e8'))
            fig.update_layout(title=f'Top values in {col}', xaxis_title='Records', yaxis_title=col, template='plotly_white', height=340, margin=dict(l=20,r=20,t=50,b=35))
            charts.append({'title':f'Category breakdown · {col}','json':fig.to_json()})

        # Strongest numeric relationships
        if len(numeric.columns) >= 2:
            corr = numeric.corr(min_periods=3)
            pairs = []
            for i, a in enumerate(corr.columns):
                for b in corr.columns[i+1:]:
                    value = corr.loc[a,b]
                    if pd.notna(value): pairs.append((abs(float(value)),str(a),str(b),float(value)))
            pairs.sort(reverse=True)
            if pairs:
                _, a, b, value = pairs[0]
                strength = 'strong' if abs(value) >= .7 else 'moderate' if abs(value) >= .4 else 'weak'
                findings.append({'type':'info','tag':'CORRELATION','title':f'{strength.title()} relationship: {a} and {b}',
                    'text':f"Pearson correlation is {value:.2f} ({'positive' if value >= 0 else 'negative'}). Correlation does not establish causation."})
                top_pairs = pairs[:8]
                fig = go.Figure(go.Bar(x=[x[3] for x in top_pairs], y=[f'{x[1]} ↔ {x[2]}' for x in top_pairs], orientation='h', marker_color=['#20a58a' if x[3] >= 0 else '#e07a65' for x in top_pairs]))
                fig.update_layout(title='Strongest numeric correlations', xaxis_title='Pearson correlation', yaxis_title='', xaxis=dict(range=[-1,1]), template='plotly_white', height=350, margin=dict(l=20,r=20,t=50,b=35))
                charts.append({'title':'Correlation analysis','json':fig.to_json()})

        if not numeric_stats and not category_summary:
            findings.append({'type':'info','tag':'DATA PROFILE','title':'Limited automatic analysis','text':'No usable numeric or low-cardinality categorical fields were found. Check column types or use the Data Cleaning page to prepare the dataset.'})
        if not recommendations:
            recommendations.append('The basic quality checks found no missing or duplicate records. Review field definitions and business context before drawing operational conclusions.')
        recommendations.append('Treat outlier flags and correlations as investigation leads; validate them with domain knowledge before making decisions.')
    return render('ai_insights.html','insights','AI Data Insights',df=df,metrics=metrics,
                  findings=findings,charts=charts,recommendations=recommendations,profile=profile)

def dataset_context(df):
    """Create a compact, privacy-conscious profile; never send the full dataset."""
    numeric = df.select_dtypes(include=np.number)
    profile = {
        'shape': {'rows': int(len(df)), 'columns': int(len(df.columns))},
        'columns': [{'name': str(c), 'dtype': str(df[c].dtype), 'missing': int(df[c].isna().sum()),
                     'unique': int(df[c].nunique(dropna=True))} for c in df.columns],
        'numeric_summary': numeric.describe().replace({np.nan: None}).round(4).to_dict() if not numeric.empty else {},
        'missing_by_column': {str(k): int(v) for k, v in df.isna().sum().items()},
        'duplicate_rows': int(df.duplicated().sum()),
        'top_categories': {}
    }
    for col in df.select_dtypes(exclude=np.number).columns[:12]:
        profile['top_categories'][str(col)] = {str(k): int(v) for k, v in df[col].astype(str).value_counts().head(8).items()}
    if len(numeric.columns) > 1:
        profile['correlations'] = numeric.corr().round(3).replace({np.nan: None}).to_dict()
    return profile


def calculate_dataset_answer(df, question):
    """Calculate common questions with Pandas before asking the LLM to explain."""
    q = question.lower()
    if any(w in q for w in ['missing', 'null', 'empty cells']):
        vals = df.isna().sum(); return f"Missing cells: {int(vals.sum())}. By column: " + ', '.join(f'{k}: {int(v)}' for k,v in vals.items() if v)
    if 'duplicate' in q: return f"There are {int(df.duplicated().sum())} duplicate rows."
    if any(w in q for w in ['how many rows','row count','number of rows','total rows']): return f"The dataset has {len(df):,} rows."
    if any(w in q for w in ['how many columns','column count','number of columns','total columns']): return f"The dataset has {len(df.columns)} columns: {', '.join(map(str,df.columns))}."
    if 'column' in q and ('list' in q or 'what are' in q or 'names' in q): return 'Columns: ' + ', '.join(map(str,df.columns))
    if 'correlation' in q or 'correlate' in q:
        n=df.select_dtypes(include=np.number)
        if n.shape[1]<2: return 'At least two numeric columns are required for correlation analysis.'
        corr=n.corr(); pairs=[]
        for i,a in enumerate(corr.columns):
            for b in corr.columns[i+1:]:
                if pd.notna(corr.loc[a,b]): pairs.append((abs(float(corr.loc[a,b])),a,b,float(corr.loc[a,b])))
        if not pairs: return 'No valid numeric correlations could be calculated.'
        pairs.sort(reverse=True); return 'Strongest numeric relationships: ' + '; '.join(f'{a} vs {b}: {v:.3f}' for _,a,b,v in pairs[:5])
    col = next((c for c in sorted(df.columns,key=lambda x:len(str(x)),reverse=True) if str(c).lower() in q), None)
    if col is None:
        return None
    if pd.api.types.is_numeric_dtype(df[col]):
        values=pd.to_numeric(df[col],errors='coerce').dropna()
        if values.empty: return f'Column {col} has no usable numeric values.'
        if any(w in q for w in ['average','mean','avg']): return f"Average {col}: {values.mean():,.4f} (based on {len(values):,} non-missing values)."
        if any(w in q for w in ['median']): return f"Median {col}: {values.median():,.4f}."
        if any(w in q for w in ['highest','maximum','max','largest','top']):
            idx=values.idxmax(); return f"Highest {col}: {values.loc[idx]:,.4f} (row index {idx})."
        if any(w in q for w in ['lowest','minimum','min','smallest']):
            idx=values.idxmin(); return f"Lowest {col}: {values.loc[idx]:,.4f} (row index {idx})."
        return f"Summary for {col}: count {len(values):,}, mean {values.mean():,.4f}, median {values.median():,.4f}, min {values.min():,.4f}, max {values.max():,.4f}, standard deviation {values.std():,.4f}."
    counts=df[col].fillna('Missing').astype(str).value_counts().head(10)
    if any(w in q for w in ['highest','most','top','best','popular','frequent']):
        return f"Top values in {col}: " + ', '.join(f'{k} ({v:,})' for k,v in counts.items())
    return f"Column {col} has {df[col].nunique(dropna=True):,} distinct non-missing values. Most common: " + ', '.join(f'{k} ({v:,})' for k,v in counts.head(5).items())


def generate_chat_reply(df, question, history):
    calculated = calculate_dataset_answer(df, question)
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            profile = dataset_context(df)
            messages = [{"role":"system","content":(
                "You are InsightForge AI, a helpful data-analysis assistant. Answer in clear, beginner-friendly language. "
                "Use ONLY the supplied dataset profile and calculated result as evidence. Never invent values or claim to have inspected raw rows. "
                "If the profile/calculation does not answer the question, say what is missing and suggest a supported question. "
                "Explain statistics briefly and mention limitations where relevant. Do not generate or execute code.\nDATASET_PROFILE=" + json.dumps(profile, ensure_ascii=False) +
                "\nCALCULATED_RESULT=" + (calculated or 'No deterministic calculation matched this question.')
            )}]
            for item in history[-4:]:
                if item.get('role') in ('user','assistant') and isinstance(item.get('content'),str):
                    messages.append({'role':item['role'],'content':item['content'][:2000]})
            messages.append({'role':'user','content':question[:2000]})
            response=client.chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'),messages=messages,temperature=0.2,max_tokens=600)
            return response.choices[0].message.content.strip(), 'AI assistant'
        except Exception as exc:
            app.logger.warning('LLM request failed: %s', exc)
            if calculated:
                return calculated + "\n\n(LLM service is currently unavailable, so this answer is based on a direct Pandas calculation.)", 'Pandas analysis'
            return "I couldn't connect to the AI service. Check your API key, internet connection, and model setting. You can still ask direct questions about column statistics, missing values, duplicates, and correlations.", 'Offline assistant'
    if calculated:
        return calculated + "\n\nTip: add OPENAI_API_KEY to your .env file to enable conversational AI explanations. The current answer was calculated directly from your dataset.", 'Pandas analysis'
    return ("I can help you explore this dataset. Try asking: ‘What is the average Sales?’, ‘Which region has the most records?’, "
            "‘Show missing values’, ‘Which columns are correlated?’, or ‘How many rows are there?’\n\n"
            "For open-ended conversational answers, configure OPENAI_API_KEY in your .env file."), 'Offline assistant'


@app.route('/ask-ai',methods=['GET'])
def ask_ai_page():
    df=get_df()
    if 'chat_history' not in session: session['chat_history']=[]
    return render('ask_ai.html','ask_ai','Ask Your Data',df=df,chat_history=session.get('chat_history',[]),ai_enabled=bool(os.getenv('OPENAI_API_KEY','').strip()))


@app.route('/ask-ai/chat',methods=['POST'])
def ask_ai_chat():
    df=get_df()
    if df is None: return {'error':'Upload a dataset first.'},400
    payload=request.get_json(silent=True) or {}
    question=str(payload.get('message','')).strip()
    if not question: return {'error':'Please enter a question.'},400
    if len(question)>2000: return {'error':'Please keep your question under 2,000 characters.'},400
    history=session.get('chat_history',[])
    answer, source=generate_chat_reply(df,question,history)
    history.extend([{'role':'user','content':question},{'role':'assistant','content':answer,'source':source}])
    session['chat_history']=history[-4:]
    session.modified=True
    return {'answer':answer,'source':source}


@app.route('/ask-ai/clear',methods=['POST'])
def clear_chat():
    session.pop('chat_history',None)
    return {'ok':True}

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

            data = df.dropna(subset=[target]).copy()
            if len(data) < 12:
                raise ValueError('At least 12 rows with a non-missing target are required.')

            y = data[target]
            if task == 'regression':
                y = pd.to_numeric(y, errors='coerce')
                valid = y.notna() & np.isfinite(y)
                data, y = data.loc[valid].copy(), y.loc[valid]
                if len(data) < 12:
                    raise ValueError('Regression needs at least 12 valid numeric target values.')
            else:
                y = y.astype(str)
                counts = y.value_counts()
                if len(counts) < 2 or counts.min() < 2:
                    raise ValueError('Classification needs at least two classes with two rows in each class.')

            X = data.drop(columns=[target]).copy()
            # Convert date-like columns to useful numeric components for ML.
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

            model_options = {
                'random_forest': (RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1) if task == 'regression' else RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)),
                'extra_trees': (ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1) if task == 'regression' else ExtraTreesClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)),
                'decision_tree': (DecisionTreeRegressor(max_depth=6, random_state=42) if task == 'regression' else DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced')),
                'linear': LinearRegression() if task == 'regression' else LogisticRegression(max_iter=1000, class_weight='balanced')
            }
            if algorithm not in model_options:
                raise ValueError('Choose a valid machine-learning algorithm.')
            model = model_options[algorithm]

            # Keep enough examples in both sets for classification classes.
            stratify = y if task == 'classification' else None
            test_size = 0.2
            if task == 'classification':
                test_count = max(int(np.ceil(len(y) * test_size)), y.nunique())
                if len(y) - test_count < y.nunique():
                    raise ValueError('Not enough rows to place every class in both training and test sets. Add more examples per class.')
                test_size = test_count
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify
            )
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
                compare.update_layout(title='Actual vs Predicted (test samples)', barmode='group', xaxis_title='Test sample', yaxis_title=target, template='plotly_white', height=390, margin=dict(l=20,r=20,t=55,b=35))
                charts.append({'title': 'Prediction comparison', 'json': compare.to_json()})
                residuals = np.asarray(y_test, dtype=float) - np.asarray(predictions, dtype=float)
                residual_fig = go.Figure(go.Histogram(x=residuals, nbinsx=20, marker_color='#4f7cff'))
                residual_fig.update_layout(title='Prediction error distribution', xaxis_title='Actual − predicted', yaxis_title='Number of rows', template='plotly_white', height=330, margin=dict(l=20,r=20,t=55,b=35))
                charts.append({'title': 'Prediction errors', 'json': residual_fig.to_json()})
                note = f'{type(model).__name__} evaluated on {len(X_test)} unseen test rows. Lower MAE/RMSE are generally better; R² can be negative and should be interpreted in context.'
            else:
                metrics = {
                    'Accuracy': round(float(accuracy_score(y_test, predictions)), 4),
                    'Precision (weighted)': round(float(precision_score(y_test, predictions, average='weighted', zero_division=0)), 4),
                    'Recall (weighted)': round(float(recall_score(y_test, predictions, average='weighted', zero_division=0)), 4),
                    'F1 (weighted)': round(float(f1_score(y_test, predictions, average='weighted', zero_division=0)), 4)
                }
                labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
                actual_counts = pd.Series(y_test).astype(str).value_counts()
                predicted_counts = pd.Series(predictions).astype(str).value_counts()
                class_fig = go.Figure()
                class_fig.add_trace(go.Bar(name='Actual', x=labels, y=[int(actual_counts.get(label, 0)) for label in labels]))
                class_fig.add_trace(go.Bar(name='Predicted', x=labels, y=[int(predicted_counts.get(label, 0)) for label in labels]))
                class_fig.update_layout(title='Actual vs Predicted class counts', barmode='group', xaxis_title='Class', yaxis_title='Rows', template='plotly_white', height=380, margin=dict(l=20,r=20,t=55,b=35))
                charts.append({'title': 'Class comparison', 'json': class_fig.to_json()})
                cm = confusion_matrix(y_test, predictions, labels=labels)
                cm_fig = go.Figure(go.Heatmap(z=cm.tolist(), x=labels, y=labels, colorscale='Blues', text=cm.tolist(), texttemplate='%{text}'))
                cm_fig.update_layout(title='Confusion matrix', xaxis_title='Predicted label', yaxis_title='Actual label', template='plotly_white', height=380, margin=dict(l=20,r=20,t=55,b=35))
                charts.append({'title': 'Confusion matrix', 'json': cm_fig.to_json()})
                note = f'{type(model).__name__} evaluated on {len(X_test)} unseen test rows. Weighted metrics account for class sizes; check the confusion matrix for class-specific errors.'

            # Tree-based feature importance, when supported by the chosen estimator.
            estimator = pipeline.named_steps['model']
            if hasattr(estimator, 'feature_importances_'):
                try:
                    names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                    importances = estimator.feature_importances_
                    top_idx = np.argsort(importances)[-12:]
                    imp_fig = go.Figure(go.Bar(x=importances[top_idx].tolist(), y=[str(names[i]).replace('numeric__','').replace('categorical__','') for i in top_idx], orientation='h', marker_color='#4f7cff'))
                    imp_fig.update_layout(title='Most influential model features', xaxis_title='Feature importance', yaxis_title='Feature', template='plotly_white', height=380, margin=dict(l=20,r=20,t=55,b=35))
                    charts.append({'title': 'Feature importance', 'json': imp_fig.to_json()})
                except Exception:
                    pass

            result = {'metrics': metrics, 'note': note, 'charts': charts, 'algorithm': type(model).__name__, 'target': target, 'task': task}
        except Exception as e:
            app.logger.exception('Prediction training failed')
            result = {'error': str(e)}
    return render('prediction.html', 'prediction', 'Machine Learning', df=df,
                  columns=df.columns.tolist() if df is not None else [], result=result)

def _report_table(rows, widths=None, header=True):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign='LEFT')
    commands = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17345f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('LEADING', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), .35, colors.HexColor('#dce4ef')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]
    if len(rows) > 1:
        for row in range(1, len(rows)):
            if row % 2 == 0:
                commands.append(('BACKGROUND', (0, row), (-1, row), colors.HexColor('#f3f6fb')))
    table.setStyle(TableStyle(commands))
    return table


def _make_report_chart(df, kind='missing'):
    """Create a small report-ready chart without saving user data to a public path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=140)
    if kind == 'missing':
        values = df.isna().sum().sort_values(ascending=False).head(10)
        values = values[values > 0]
        if values.empty:
            plt.close(fig)
            return None
        ax.bar([str(x)[:22] for x in values.index], values.values)
        ax.set_title('Missing values by column', loc='left', fontsize=11, weight='bold')
        ax.set_ylabel('Missing cells')
        ax.tick_params(axis='x', rotation=25, labelsize=8)
    elif kind == 'numeric':
        numeric = df.select_dtypes(include=np.number)
        if numeric.empty:
            plt.close(fig)
            return None
        cols = numeric.columns[:8]
        means = [pd.to_numeric(numeric[c], errors='coerce').mean() for c in cols]
        ax.bar([str(x)[:18] for x in cols], means)
        ax.set_title('Average value by numeric field', loc='left', fontsize=11, weight='bold')
        ax.set_ylabel('Mean')
        ax.tick_params(axis='x', rotation=25, labelsize=8)
    fig.tight_layout()
    image = BytesIO()
    fig.savefig(image, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    image.seek(0)
    return image


def build_professional_pdf(df, dataset_name, output_path):
    """Build a structured, data-driven PDF report using the active dataframe."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CoverTitleIF', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=25, leading=30, textColor=colors.HexColor('#17345f'), alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name='SubIF', parent=styles['Normal'], fontSize=10, leading=15, textColor=colors.HexColor('#66758d'), spaceAfter=12))
    styles.add(ParagraphStyle(name='H1IF', parent=styles['Heading1'], fontSize=15, leading=19, textColor=colors.HexColor('#17345f'), spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle(name='H2IF', parent=styles['Heading2'], fontSize=10, leading=13, textColor=colors.HexColor('#3268e8'), spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name='BodyIF', parent=styles['BodyText'], fontSize=8.5, leading=13, textColor=colors.HexColor('#35445c'), spaceAfter=6))
    styles.add(ParagraphStyle(name='SmallIF', parent=styles['BodyText'], fontSize=7.5, leading=10, textColor=colors.HexColor('#66758d')))
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=17*mm, leftMargin=17*mm, topMargin=18*mm, bottomMargin=17*mm, title='InsightForge AI Data Analysis Report', author='InsightForge AI')
    story=[]
    rows, cols = df.shape
    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())
    cell_total = max(rows*cols, 1)
    score = max(0, round(100*(1-(missing_total+duplicate_total*max(cols,1))/cell_total)))
    numeric = df.select_dtypes(include=np.number)
    categorical = [c for c in df.columns if c not in numeric.columns and df[c].nunique(dropna=True) <= 30]

    story += [Spacer(1, 16*mm), Paragraph('INSIGHTFORGE AI', styles['H2IF']),
              Paragraph('Data Analysis Report', styles['CoverTitleIF']),
              Paragraph('An evidence-based summary of dataset structure, quality, descriptive statistics and observed patterns.', styles['SubIF']),
              Spacer(1, 5*mm)]
    meta = [['DATASET', html.escape(str(dataset_name)[:90])], ['GENERATED', datetime.now().strftime('%d %B %Y, %H:%M')], ['ANALYSIS SCOPE', 'Uploaded dataset · descriptive analysis']]
    story.append(_report_table(meta, widths=[38*mm, 125*mm], header=False))
    story += [Spacer(1, 8*mm), Paragraph('Executive summary', styles['H1IF']),
              Paragraph(f'The uploaded dataset contains <b>{rows:,} records</b> across <b>{cols:,} fields</b>. The automated quality indicator is <b>{score}/100</b>. The analysis detected <b>{missing_total:,} missing cells</b> and <b>{duplicate_total:,} duplicate rows</b>. This report describes patterns present in the supplied data; it does not establish causation or guarantee predictive performance.', styles['BodyIF'])]

    story.append(Paragraph('1. Dataset overview', styles['H1IF']))
    overview=[['Metric','Value'],['Rows',f'{rows:,}'],['Columns',f'{cols:,}'],['Numeric fields',str(len(numeric.columns))],['Categorical / other fields',str(cols-len(numeric.columns))],['Memory usage',f'{df.memory_usage(deep=True).sum()/1024:.1f} KB']]
    story.append(_report_table(overview, widths=[75*mm, 88*mm]))

    story.append(Paragraph('2. Data quality assessment', styles['H1IF']))
    qrows=[['Quality check','Result'],['Data quality indicator',f'{score}/100 (heuristic)'],['Missing cells',f'{missing_total:,}'],['Duplicate rows',f'{duplicate_total:,}'],['Columns containing missing values',str(int((df.isna().sum()>0).sum()))]]
    story.append(_report_table(qrows, widths=[75*mm,88*mm]))
    if missing_total:
        story.append(Paragraph('Missing values by field', styles['H2IF']))
        miss=df.isna().sum().sort_values(ascending=False)
        mrows=[['Column','Missing','Missing %']]
        for c,v in miss[miss>0].head(20).items(): mrows.append([Paragraph(html.escape(str(c)),styles['SmallIF']),f'{int(v):,}',f'{(v/max(rows,1))*100:.1f}%'])
        story.append(_report_table(mrows,widths=[85*mm,35*mm,43*mm]))
        img=_make_report_chart(df,'missing')
        if img: story += [Spacer(1,4*mm),RLImage(img,width=160*mm,height=62*mm)]
    else:
        story.append(Paragraph('No missing cells were detected in the uploaded dataset.', styles['BodyIF']))

    story.append(Paragraph('3. Descriptive statistics', styles['H1IF']))
    if not numeric.empty:
        desc=numeric.describe().T.replace([np.inf,-np.inf],np.nan).round(3)
        stat_rows=[['Numeric field','Count','Mean','Std. dev.','Min','Median','Max']]
        for col,row in desc.iterrows():
            median=pd.to_numeric(df[col],errors='coerce').median()
            vals=[row.get('count'),row.get('mean'),row.get('std'),row.get('min'),median,row.get('max')]
            stat_rows.append([Paragraph(html.escape(str(col)[:35]),styles['SmallIF'])]+[('—' if pd.isna(v) else f'{float(v):,.3f}') for v in vals])
        story.append(_report_table(stat_rows,widths=[42*mm,18*mm,23*mm,23*mm,19*mm,19*mm,19*mm]))
        img=_make_report_chart(df,'numeric')
        if img: story += [Spacer(1,4*mm),RLImage(img,width=160*mm,height=62*mm)]
    else:
        story.append(Paragraph('No numeric fields were detected; descriptive numeric statistics are not applicable.',styles['BodyIF']))

    story.append(Paragraph('4. Categorical profile', styles['H1IF']))
    if categorical:
        for col in categorical[:8]:
            counts=df[col].fillna('Missing').astype(str).value_counts().head(5)
            story.append(Paragraph(html.escape(str(col)),styles['H2IF']))
            crows=[['Category','Records']]+[[Paragraph(html.escape(str(k)[:70]),styles['SmallIF']),f'{int(v):,}'] for k,v in counts.items()]
            story.append(_report_table(crows,widths=[125*mm,38*mm]))
    else:
        story.append(Paragraph('No low-cardinality categorical fields were available for a compact category summary.',styles['BodyIF']))

    story.append(Paragraph('5. Correlation analysis', styles['H1IF']))
    if len(numeric.columns)>=2:
        corr=numeric.corr(numeric_only=True)
        pairs=[]
        for i,a in enumerate(corr.columns):
            for b in corr.columns[i+1:]:
                value=corr.loc[a,b]
                if pd.notna(value): pairs.append((abs(float(value)),str(a),str(b),float(value)))
        pairs.sort(reverse=True)
        if pairs:
            crows=[['Field A','Field B','Pearson r','Direction']]
            for _,a,b,v in pairs[:10]: crows.append([Paragraph(html.escape(a[:35]),styles['SmallIF']),Paragraph(html.escape(b[:35]),styles['SmallIF']),f'{v:.3f}','Positive' if v>0 else 'Negative' if v<0 else 'None'])
            story.append(_report_table(crows,widths=[48*mm,48*mm,28*mm,39*mm]))
            story.append(Paragraph('Correlation describes linear association only. It does not prove that one field causes changes in another.',styles['SmallIF']))
        else: story.append(Paragraph('No valid pairwise numeric correlations could be calculated.',styles['BodyIF']))
    else: story.append(Paragraph('At least two numeric fields are required for correlation analysis.',styles['BodyIF']))

    story.append(Paragraph('6. Data dictionary', styles['H1IF']))
    drows=[['Field','Data type','Non-missing','Unique']]
    for col in df.columns[:40]: drows.append([Paragraph(html.escape(str(col)[:40]),styles['SmallIF']),str(df[col].dtype),f'{int(df[col].notna().sum()):,}',f'{int(df[col].nunique(dropna=True)):,}'])
    story.append(_report_table(drows,widths=[65*mm,35*mm,32*mm,31*mm]))
    if cols>40: story.append(Paragraph(f'{cols-40} additional fields omitted from this compact data dictionary.',styles['SmallIF']))

    story.append(Paragraph('7. Recommendations and limitations', styles['H1IF']))
    recommendations=[]
    if missing_total: recommendations.append('Review fields with missing values and select an imputation or exclusion strategy appropriate to the meaning of each field.')
    if duplicate_total: recommendations.append('Inspect duplicate records and confirm whether they represent repeated events or accidental duplicates before removing them.')
    if not missing_total and not duplicate_total: recommendations.append('Basic missing-value and duplicate checks did not identify issues; continue with domain-specific validation and business rules.')
    if numeric.empty: recommendations.append('Review column data types if numeric measures are expected but were imported as text.')
    recommendations += ['Investigate unusual values and correlations with subject-matter knowledge before taking action.', 'For machine-learning work, select a suitable target, avoid data leakage, and evaluate on unseen data.']
    for item in recommendations: story.append(Paragraph('• '+html.escape(item),styles['BodyIF']))
    story.append(Paragraph('Limitations: the quality indicator is a simple heuristic; descriptive patterns depend on the uploaded sample; correlation is not causation; this report does not independently validate source data or imply that a predictive model has been trained.',styles['SmallIF']))

    def page_decor(canvas_obj, doc_obj):
        canvas_obj.saveState(); w,h=A4
        if doc_obj.page>1:
            canvas_obj.setStrokeColor(colors.HexColor('#e2e8f0')); canvas_obj.line(17*mm,h-12*mm,w-17*mm,h-12*mm)
            canvas_obj.setFont('Helvetica',7); canvas_obj.setFillColor(colors.HexColor('#7b8798')); canvas_obj.drawString(17*mm,h-9*mm,'INSIGHTFORGE AI  |  DATA ANALYSIS REPORT')
        canvas_obj.setStrokeColor(colors.HexColor('#e2e8f0')); canvas_obj.line(17*mm,12*mm,w-17*mm,12*mm)
        canvas_obj.setFont('Helvetica',7); canvas_obj.setFillColor(colors.HexColor('#7b8798')); canvas_obj.drawString(17*mm,8*mm,'Confidential · Generated from the active uploaded dataset')
        canvas_obj.drawRightString(w-17*mm,8*mm,f'Page {doc_obj.page}')
        canvas_obj.restoreState()
    doc.build(story,onFirstPage=page_decor,onLaterPages=page_decor)


@app.route('/reports',methods=['GET','POST'])
def reports_page():
    df=get_df(); generated=False
    report_name=session.get('report_file')
    if request.method=='POST':
        if df is None:
            flash('Upload a dataset before generating a report.','error')
        else:
            try:
                stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
                report_name=f'InsightForge_Report_{stamp}.pdf'
                path=os.path.join(REPORT_FOLDER,report_name)
                build_professional_pdf(df,session.get('dataset_name','Uploaded dataset'),path)
                session['report_file']=report_name
                generated=True
                flash('Professional PDF report generated successfully.','success')
            except Exception as exc:
                app.logger.exception('Report generation failed')
                flash(f'Could not generate report: {exc}','error')
    elif report_name and os.path.isfile(os.path.join(REPORT_FOLDER,secure_filename(report_name))):
        generated=True
    return render('reports.html','reports','Reports',df=df,generated=generated,report_name=report_name,
                  stats=quality(df) if df is not None else None,
                  numeric_count=len(df.select_dtypes(include=np.number).columns) if df is not None else 0)


@app.route('/reports/download')
def download_report():
    name=secure_filename(session.get('report_file',''))
    if not name or not name.startswith('InsightForge_Report_') or not name.endswith('.pdf'):
        abort(404)
    path=os.path.join(REPORT_FOLDER,name)
    if not os.path.isfile(path): abort(404)
    return send_file(path,as_attachment=True,download_name=name,mimetype='application/pdf')

@app.route('/settings',methods=['GET','POST'])
def settings_page():
    if request.method=='POST': session.clear(); flash('Active dataset session cleared. Uploaded files remain in the local uploads folder.','success'); return redirect(url_for('settings_page'))
    return render('settings.html','settings','Settings')

@app.errorhandler(413)
def too_large(_): flash('File exceeds the 25 MB upload limit.','error'); return redirect(url_for('upload_page'))

if __name__=='__main__': app.run(debug=True)

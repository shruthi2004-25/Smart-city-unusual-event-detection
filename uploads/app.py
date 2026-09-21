import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Required for Flask to prevent GUI errors
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'smart_city_secret'
db = SQLAlchemy(app)

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

# --- Helper Functions ---
def generate_data_logic():
    """Generates synthetic surveillance data for testing."""
    rows = 100
    np.random.seed(42)
    data = {
        'timestamp_hour': np.random.randint(0, 24, rows),
        'traffic_density': np.random.normal(45, 8, rows),
        'noise_level': np.random.normal(55, 4, rows),
        'pedestrian_count': np.random.normal(150, 25, rows),
        'emergency_vehicle_siren': np.zeros(rows)
    }
    df = pd.DataFrame(data)
    
    # Inject specific anomalies
    df.loc[10:15, 'traffic_density'] = 110
    df.loc[10:15, 'noise_level'] = 120
    
    # Clip numeric data to ensure no negatives
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
    
    return df

def detect_unusual_events(df):
    """Isolation Forest to detect anomalies."""
    # Work on a copy to avoid modifying the original dataframe globally
    df_processed = df.copy()
    
    # Select only numeric columns for the model
    model_data = df_processed.select_dtypes(include=[np.number])
    
    model = IsolationForest(contamination=0.1, random_state=42)
    df_processed['anomaly_score'] = model.fit_predict(model_data)
    
    # -1 is anomaly, 1 is normal
    df_processed['is_unusual'] = df_processed['anomaly_score'].apply(
        lambda x: 'Unusual' if x == -1 else 'Normal'
    )
    return df_processed

def generate_visuals(df):
    """Generates and saves plots for the dashboard."""
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')
    
    # Plot 1: Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_unusual', data=df, palette='viridis')
    plt.title('Detected Unusual Events')
    plt.tight_layout()
    plt.savefig('static/plots/distribution.png')
    plt.close()

    # Plot 2: Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Map')
    plt.tight_layout()
    plt.savefig('static/plots/heatmap.png')
    plt.close()

# --- Routes ---
@app.route('/')
def home():
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        new_user = User(username=username, password=password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Registration successful! Proceed to Dashboard.")
            return redirect(url_for('upload'))
        except IntegrityError:
            db.session.rollback()
            flash("Error: That username is already taken.")
            return redirect(url_for('signup'))
    return render_template('signup.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'generate' in request.form:
            df = generate_data_logic()
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'synthetic_data.csv')
            df.to_csv(file_path, index=False)
        else:
            file = request.files.get('file')
            if file and file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                df = pd.read_csv(file_path)
            else:
                flash("No file selected for upload.")
                return redirect(url_for('upload'))

        # Process, Visualize, and Display
        results_df = detect_unusual_events(df)
        generate_visuals(results_df)
        
        # Pass Top 10 rows to dashboard
        table_html = results_df.head(10).to_html(classes='data', index=False, escape=False)
        return render_template('dashboard.html', table=table_html)
    
    return render_template('upload.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
from flask import Flask, request, render_template, send_file
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = Flask(__name__)

# Load the diabetic retinopathy model
dr_model = tf.keras.models.load_model('dr_severity_model.h5')
DR_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Load the diabetes prediction model (replace with your model)
diabetes_model = joblib.load('diabetes_model.pkl')

# Global variables to store diabetes prediction results and dataset
results = []
dataset = None

# Function to preprocess an image for diabetic retinopathy
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to preprocess CSV data for diabetes prediction
def preprocess_csv(csv_path):
    global dataset
    df = pd.read_csv(csv_path)
    
    # Debug: Print the columns in the CSV file
    print("Columns in the CSV file:", df.columns.tolist())
    
    # Drop extra columns (e.g., index or target column)
    df = df.drop(columns=['Outcome'], errors='ignore')  # Adjust column names as needed
    
    # Ensure the CSV has the correct number of features
    if df.shape[1] != 8:  # Replace 8 with the number of features expected by your model
        raise ValueError(f"Expected 8 features, but found {df.shape[1]} features in the CSV file.")
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Store the dataset for EDA
    dataset = df
    return scaled_data

# Function to generate EDA insights and visualizations
def generate_eda(dataset):
    insights = {}
    visualizations = {}

    # Summary statistics
    insights['summary'] = dataset.describe().to_html()

    # Correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    correlation_img = BytesIO()
    plt.savefig(correlation_img, format='png')
    plt.close()
    correlation_img.seek(0)
    visualizations['correlation'] = base64.b64encode(correlation_img.getvalue()).decode('utf-8')

    # Histograms for numerical columns
    histograms = {}
    for column in dataset.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(dataset[column], kde=True)
        plt.title(f'Distribution of {column}')
        hist_img = BytesIO()
        plt.savefig(hist_img, format='png')
        plt.close()
        hist_img.seek(0)
        histograms[column] = base64.b64encode(hist_img.getvalue()).decode('utf-8')
    visualizations['histograms'] = histograms

    return insights, visualizations

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict_dr', methods=['POST'])
def predict_dr():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No image selected", 400

    # Save the uploaded image
    upload_dir = 'static/uploads'
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, file.filename)
    file.save(image_path)

    # Preprocess the image and predict
    img = preprocess_image(image_path)
    predictions = dr_model.predict(img)
    severity = DR_LABELS[np.argmax(predictions)]  # Get the predicted class
    confidence = float(np.max(predictions)) * 100  # Confidence score

    return render_template('result.html', image_path=image_path, severity=severity, confidence=confidence)

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    global results, dataset
    
    if 'csv' not in request.files:
        return "No CSV file uploaded", 400

    file = request.files['csv']
    if file.filename == '':
        return "No CSV file selected", 400

    # Save the uploaded CSV
    upload_dir = 'static/uploads'
    os.makedirs(upload_dir, exist_ok=True)
    csv_path = os.path.join(upload_dir, file.filename)
    file.save(csv_path)

    # Preprocess the CSV and predict
    data = preprocess_csv(csv_path)
    predictions = diabetes_model.predict(data)  # Predict diabetes
    results = ["Diabetic" if pred == 1 else "Non-Diabetic" for pred in predictions]

    # Generate EDA insights and visualizations
    insights, visualizations = generate_eda(dataset)

    # Pagination: Display first 20 rows
    page = request.args.get('page', 1, type=int)
    per_page = 20
    start = (page - 1) * per_page
    end = start + per_page
    paginated_results = results[start:end]

    return render_template('result_diabetes.html', results=paginated_results, page=page, insights=insights, visualizations=visualizations)

@app.route('/download_results')
def download_results():
    # Create a CSV file in memory
    output = io.StringIO()
    output.write("Row,Prediction\n")
    for i, result in enumerate(results):
        output.write(f"{i + 1},{result}\n")
    
    # Prepare the file for download
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='diabetes_predictions.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)
pip install flask opencv-python numpy scikit-image scikit-learn


from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

app = Flask(__name__)

rf_model = pickle.load(open("random_forest_spiral.pkl", "rb"))
knn_model = pickle.load(open("knn_wave.pkl", "rb"))

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 50, 150)
    hog_features = hog(edges, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return np.array(hog_features).reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    category = request.form['category']  # Spiral or Wave
    
    if file.filename == '':
        return "No selected file"
    
    file_path = os.path.join("static/uploads", file.filename)
    file.save(file_path)
    
    features = preprocess_image(file_path)
    
    if category == "spiral":
        model = rf_model
    else:
        model = knn_model
    
    prediction = model.predict(features)
    
    result = "Parkinson's Detected" if prediction[0] == 1 else "Healthy"

    return render_template("index.html", prediction=result, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)

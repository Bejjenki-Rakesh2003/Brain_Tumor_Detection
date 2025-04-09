from flask import Flask, render_template, request, redirect, url_for, session, make_response
import mysql.connector
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model
model_path = "D:\\brain_tumor_detection\\brain_tumor_detection_model.h5"
model = load_model(model_path)
img_height, img_width = 128, 128

# Allowed extensions for image validation
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

def get_db_connection():
    return mysql.connector.connect(host="localhost", user="root", password="aishu", database="user_db")

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db_connection()
        cursor = db.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            db.commit()
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            return render_template('register.html', message="Username already exists.")
        finally:
            cursor.close()
            db.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        db.close()
        if result and result[0] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', message="Invalid credentials")
    return render_template('login.html')

@app.route('/index')
def index():
    if 'username' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if 'image' not in request.files:
        return render_template('predict.html', prediction="No image file provided", image_path=None)

    image = request.files['image']
    if image.filename == '':
        return render_template('predict.html', prediction="No selected file", image_path=None)

    file_ext = image.filename.rsplit('.', 1)[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return render_template('predict.html', prediction="Invalid Image Submission", image_path=None)

    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        return render_template('predict.html', prediction="Invalid Image Submission", image_path=None)

    img_cv = cv2.resize(img_cv, (img_height, img_width))
    img_array = np.array(img_cv) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    prediction = model.predict(img_array)
    result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
    
    # Path relative to /static
    image_url = f"uploads/{filename}"

    return render_template('predict.html', prediction=result, image_path=image_url)

@app.route('/logout')
def logout():
    session.clear()
    response = make_response(redirect(url_for('login')))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    app.run(debug=True)

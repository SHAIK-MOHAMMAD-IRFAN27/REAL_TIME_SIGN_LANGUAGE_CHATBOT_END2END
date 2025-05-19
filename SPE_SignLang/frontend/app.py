from flask import Flask, render_template, request, redirect, url_for, session
import requests
import os
import uuid
import socket
import json
import logging

app = Flask(__name__, static_folder='static', static_url_path='/static')
# app.secret_key = '09edfe6d16b0c1faad53e2d1b0b235fbc942142bb25ccdac5e877e9dce73c202' 
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'fallback_dev_key')

def setup_logger():
    log_dir = '/var/log/app'
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('frontend-service')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(f'{log_dir}/frontend.log', mode='a')  # Changed to frontend.log
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                "service": "frontend-service",
                "level": record.levelname.lower(),
                "message": record.getMessage(),
                "host": socket.gethostname()
            }
            return json.dumps(log_record)
    
    formatter = JSONFormatter()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Logger configured successfully")
    return logger

logger = setup_logger()

upload_folder = 'static/uploads'
os.makedirs(upload_folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder

PREDICTION_SERVICE_URL = os.environ.get('PREDICTION_SERVICE_URL', 'http://backend:5000')
LOGIN_SERVICE_URL = os.environ.get('LOGIN_SERVICE_URL', 'http://login:5002')

@app.route('/')
def index():
    username = session.get('username')
    if not username:
        logger.info("User not logged in, redirecting to login page")
        return redirect(f"{LOGIN_SERVICE_URL}/login")
    logger.info(f"User {username} accessed index page")
    return render_template('index.html', username=username)

@app.route('/logout')
def logout():
    username = session.get('username')
    logger.info(f"User {username} logged out")
    session.clear()
    return redirect(f"{LOGIN_SERVICE_URL}/login")

@app.route('/predict', methods=['POST'])
def predict():
    username = session.get('username')
    if not username:
        logger.warning("Unauthorized prediction attempt - no user session")
        return redirect("/login")

    if 'file' not in request.files or request.files['file'].filename == '':
        logger.warning(f"User {username} attempted prediction without file")
        return redirect(url_for('index'))

    file = request.files['file']
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    logger.info(f"User {username} uploaded file: {filename}")
    file.save(filepath)

    try:
        logger.info(f"User {username} requesting prediction for {filename}")
        with open(filepath, 'rb') as f:
            response = requests.post(f"{PREDICTION_SERVICE_URL}/predict", files={'file': f})

        if response.status_code == 200:
            data = response.json()
            predicted_letter = data.get("prediction_letter", "Unknown")
            logger.info(f"Prediction successful for user {username} - Letter: {predicted_letter}")
            return render_template('index.html',
                                   prediction=predicted_letter,
                                   image_path=f"/static/uploads/{filename}",
                                   username=username)
        else:
            logger.error(f"Prediction service error for user {username} - Status code: {response.status_code}")
            return render_template('index.html',
                                   error="Prediction service error.",
                                   username=username)
    except Exception as e:
        logger.error(f"Error processing prediction for user {username}: {str(e)}", exc_info=True)
        return render_template('index.html',
                               error=f"Error: {str(e)}",
                               username=username)

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    logger.info("Starting frontend service on port 5001")
    try:
        app.run(host='0.0.0.0', port=5001, debug=True)
    except Exception as e:
        logger.critical(f"Service failed to start: {str(e)}", exc_info=True)
        raise

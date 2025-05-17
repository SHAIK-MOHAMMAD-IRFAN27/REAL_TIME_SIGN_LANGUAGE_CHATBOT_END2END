# from flask import Flask, render_template, request, redirect, url_for
# import requests
# import os
# import uuid

# app = Flask(__name__, static_folder='static', static_url_path='/static')

# upload_folder = 'static/uploads'
# if not os.path.exists(upload_folder):
#     os.makedirs(upload_folder)

# app.config['UPLOAD_FOLDER'] = upload_folder

# # Point to the backend prediction service (Kubernetes DNS name)
# PREDICTION_SERVICE_URL = os.environ.get('PREDICTION_SERVICE_URL', 'http://backend:5000')

# @app.route('/')
# def index():
#     username = request.args.get('user', None)
#     if not username:
#         return redirect("/login")  # Redirect to login page via Ingress
#     return render_template('index.html', username=username)

# @app.route('/logout')
# def logout():
#     return redirect("http://signlanguage.local/login")

# @app.route('/predict', methods=['POST'])
# def predict():
#     username = request.args.get('user', None)

#     if not username:
#         return redirect("/login")

#     if 'file' not in request.files or request.files['file'].filename == '':
#         return redirect(url_for('index', user=username))

#     file = request.files['file']
#     filename = f"{uuid.uuid4()}_{file.filename}"
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)

#     try:
#         with open(filepath, 'rb') as f:
#             response = requests.post(f"{PREDICTION_SERVICE_URL}/predict", files={'file': f})

#         if response.status_code == 200:
#             data = response.json()
#             predicted_letter = data.get("prediction_letter", "Unknown")
#             return render_template('index.html',
#                                    prediction=predicted_letter,
#                                    image_path=f"/static/uploads/{filename}",
#                                    username=username)
#         else:
#             return render_template('index.html',
#                                    error="Prediction service error.",
#                                    username=username)
#     except Exception as e:
#         return render_template('index.html',
#                                error=f"Error: {str(e)}",
#                                username=username)

# @app.after_request
# def add_no_cache_headers(response):
#     response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
#     response.headers["Pragma"] = "no-cache"
#     response.headers["Expires"] = "0"
#     return response

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001, debug=True)


from flask import Flask, render_template, request, redirect, url_for, session
import requests
import os
import uuid

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = '09edfe6d16b0c1faad53e2d1b0b235fbc942142bb25ccdac5e877e9dce73c202' 

upload_folder = 'static/uploads'
os.makedirs(upload_folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder

PREDICTION_SERVICE_URL = os.environ.get('PREDICTION_SERVICE_URL', 'http://backend:5000')

LOGIN_SERVICE_URL = os.environ.get('LOGIN_SERVICE_URL', 'http://login:5002')

@app.route('/')
def index():
    username = session.get('username')
    if not username:
        return redirect(f"{LOGIN_SERVICE_URL}/login")
    return render_template('index.html', username=username)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(f"{LOGIN_SERVICE_URL}/login")


@app.route('/predict', methods=['POST'])
def predict():
    username = session.get('username')
    if not username:
        return redirect("/login")

    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['file']
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        with open(filepath, 'rb') as f:
            response = requests.post(f"{PREDICTION_SERVICE_URL}/predict", files={'file': f})

        if response.status_code == 200:
            data = response.json()
            predicted_letter = data.get("prediction_letter", "Unknown")
            return render_template('index.html',
                                   prediction=predicted_letter,
                                   image_path=f"/static/uploads/{filename}",
                                   username=username)
        else:
            return render_template('index.html',
                                   error="Prediction service error.",
                                   username=username)
    except Exception as e:
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
    app.run(host='0.0.0.0', port=5001, debug=True)

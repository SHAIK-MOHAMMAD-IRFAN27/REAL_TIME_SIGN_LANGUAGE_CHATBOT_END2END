
from flask import Flask, request, render_template, redirect, session
import sqlite3
import os
import socket
import json
import logging

app = Flask(__name__)
# app.secret_key = '09edfe6d16b0c1faad53e2d1b0b235fbc942142bb25ccdac5e877e9dce73c202'  # Add a secure secret key
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'fallback_dev_key')

def setup_logger():
    # Create logs directory if not exists
    log_dir = '/var/log/app'
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger('login-service')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler for JSON logs
    file_handler = logging.FileHandler(f'{log_dir}/login-service.log', mode='a')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # JSON formatter
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                "service": "login-service",
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
    
    # Test the logger
    logger.info("Logger configured successfully")
    return logger

logger = setup_logger()

DB_PATH = "/app/data/users.db"
DB_DIR = os.path.dirname(DB_PATH)


FRONTEND_URL = os.getenv("FRONTEND_URL", "http://frontend:5001/")

def init_db():
    try:
        logger.info("Initializing database...")
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
        raise

@app.route('/')
def home():
    logger.info("Login page accessed")
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        logger.warning("Registration attempt with missing credentials")
        return "Username and password required", 400

    try:
        logger.info(f"Registration attempt for user: {username}")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        logger.info(f"User {username} registered successfully")
        return redirect('/login')
    except sqlite3.IntegrityError:
        logger.warning(f"Registration failed - User {username} already exists")
        return "User already exists"
    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"
    finally:
        conn.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        logger.info(f"Login attempt for user: {username}")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        result = c.fetchone()
        conn.close()

        if result:
            session['username'] = username
            logger.info(f"User {username} logged in successfully")
            return redirect(FRONTEND_URL)  
        else:
            logger.warning(f"Failed login attempt for user: {username}")
            return "Invalid credentials"
    else:
        logger.info("Login page accessed")
        return render_template('login.html')

if __name__ == '__main__':
    logger.info("Starting login service on port 5002")
    try:
        init_db()
        app.run(host='0.0.0.0', port=5002, debug=True)
    except Exception as e:
        logger.critical(f"Service failed to start: {str(e)}", exc_info=True)
        raise

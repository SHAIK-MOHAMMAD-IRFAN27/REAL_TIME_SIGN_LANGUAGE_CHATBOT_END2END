# from flask import Flask, request, render_template, redirect, url_for
# import sqlite3
# import os

# app = Flask(__name__)


# DB_PATH = "/app/data/users.db"
# DB_DIR = os.path.dirname(DB_PATH)

# def init_db():
#     try:
#         # Ensure directory exists
#         os.makedirs(DB_DIR, exist_ok=True)

#         # Connect to SQLite (creates the DB file if it doesn't exist)
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()

#         # Create users table if not exists
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS users (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 username TEXT NOT NULL,
#                 password TEXT NOT NULL
#             )
#         ''')
#         conn.commit()
#         conn.close()
#     except Exception as e:
#         print("Failed to initialize DB:", str(e))
#         raise

# @app.route('/')
# def home():
#     return render_template('login.html')

# @app.route('/register', methods=['POST'])
# def register():
#     username = request.form.get('username')
#     password = request.form.get('password')

#     if not username or not password:
#         return "Username and password required", 400

#     try:
#         conn = sqlite3.connect(DB_PATH)
#         c = conn.cursor()
#         c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
#         conn.commit()
#         return redirect('/login')  # <- Fixed here
#     except sqlite3.IntegrityError:
#         return "User already exists"
#     except Exception as e:
#         return f"Error: {str(e)}"
#     finally:
#         conn.close()

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         # your existing login logic
#         username = request.form['username']
#         password = request.form['password']

#         conn = sqlite3.connect(DB_PATH)
#         c = conn.cursor()
#         c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
#         result = c.fetchone()
#         conn.close()

#         if result:
#             return redirect(f"http://signlanguage.local/?user={username}")
#         else:
#             return "Invalid credentials"
#     else:
#         # render login form on GET
#         return render_template('login.html')
# if __name__ == '__main__':
#     init_db()
#     app.run(host='0.0.0.0', port=5002, debug=True)


from flask import Flask, request, render_template, redirect, session
import sqlite3
import os

app = Flask(__name__)
app.secret_key = '09edfe6d16b0c1faad53e2d1b0b235fbc942142bb25ccdac5e877e9dce73c202'  # Add a secure secret key

DB_PATH = "/app/data/users.db"
DB_DIR = os.path.dirname(DB_PATH)


FRONTEND_URL = os.getenv("FRONTEND_URL", "http://frontend:5001/")

def init_db():
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

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return "Username and password required", 400

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return redirect('/login')
    except sqlite3.IntegrityError:
        return "User already exists"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        conn.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        result = c.fetchone()
        conn.close()

        if result:
            session['username'] = username
            return redirect(FRONTEND_URL)  
        else:
            return "Invalid credentials"
    else:
        return render_template('login.html')

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5002, debug=True)

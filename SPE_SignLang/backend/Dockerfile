FROM tensorflow/tensorflow:2.15.0

# Set working directory
WORKDIR /app

# Copy only requirements first (to cache dependencies if unchanged)
COPY requirements.txt .

# Install Python dependencies with --ignore-installed flag
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Flask on port 5000 (default for Flask)
EXPOSE 5000

# Run Flask application (assuming app.py is the entry point)
CMD ["python", "signlang.py"]

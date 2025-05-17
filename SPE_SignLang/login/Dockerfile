FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y sqlite3

COPY . .

EXPOSE 5002

CMD ["python", "login_service.py"]




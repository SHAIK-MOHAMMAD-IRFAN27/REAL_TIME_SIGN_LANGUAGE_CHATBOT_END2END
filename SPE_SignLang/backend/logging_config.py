import logging
import logstash
import sys
import os
import datetime

# Create logger
logger = logging.getLogger("signlang-backend")
logger.setLevel(logging.INFO)

# Prevent adding handlers multiple times
if not logger.hasHandlers():
    # Logstash handler
    try:
        logstash_handler = logstash.LogstashHandler("logstash", 5044, version=1)
        logger.addHandler(logstash_handler)
    except Exception as e:
        print("Logstash handler failed:", e)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    # File handler
    file_handler = logging.FileHandler('logs/backend.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

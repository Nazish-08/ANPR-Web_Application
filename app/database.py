import os
import mysql.connector
from mysql.connector import Error


# ============================================
# DATABASE CONFIGURATION
# ============================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "database": os.getenv("DB_NAME", "anpr_database"),
}


# ============================================
# DATABASE CONNECTION
# ============================================

def get_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as error:
        print(f"[DATABASE ERROR] {error}")
    return None


# ============================================
# SAVE DETECTION
# ============================================

def save_detection(plate_text, track_id, confidence, timestamp, source):
    connection = get_connection()
    if connection is None:
        return False

    cursor = None
    try:
        cursor = connection.cursor()
        query = """
            INSERT INTO detections (
                plate_text, track_id, confidence, timestamp, source
            ) VALUES (%s, %s, %s, %s, %s)
        """
        values = (plate_text, track_id, confidence, timestamp, source)
        cursor.execute(query, values)
        connection.commit()
        print("[INFO] Detection saved to database.")
        return True
    except Error as error:
        print(f"[DATABASE ERROR] {error}")
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if connection.is_connected():
            connection.close()


# ============================================
# GET ALL DETECTIONS
# ============================================

def get_detections():
    connection = get_connection()
    if connection is None:
        return []

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT id, plate_text, track_id, confidence, timestamp, source
            FROM detections
            ORDER BY timestamp DESC
        """
        cursor.execute(query)
        return cursor.fetchall()
    except Error as error:
        print(f"[DATABASE ERROR] {error}")
        return []
    finally:
        if cursor is not None:
            cursor.close()
        if connection.is_connected():
            connection.close()


# ============================================
# GET SINGLE DETECTION
# ============================================

def get_detection_by_id(detection_id):
    connection = get_connection()
    if connection is None:
        return None

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT id, plate_text, track_id, confidence, timestamp, source
            FROM detections
            WHERE id = %s
        """
        cursor.execute(query, (detection_id,))
        return cursor.fetchone()
    except Error as error:
        print(f"[DATABASE ERROR] {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection.is_connected():
            connection.close()
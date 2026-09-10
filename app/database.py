import mysql.connector
from mysql.connector import Error


# ============================================
# DATABASE CONFIGURATION
# ============================================

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "anpr_database",
}


# ============================================
# DATABASE CONNECTION
# ============================================

def get_connection():
    """
    Create and return a MySQL database connection.

    Returns:
        MySQL connection object if successful.
        None if connection fails.
    """

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

def save_detection(
    plate_text,
    track_id,
    confidence,
    timestamp,
    source
):
    """
    Save one ANPR detection into the detections table.

    Args:
        plate_text: Detected vehicle number plate text.
        track_id: Unique tracking ID of the vehicle.
        confidence: OCR/detection confidence.
        timestamp: Detection timestamp.
        source: Source image or video filename.

    Returns:
        True if successfully saved.
        False if insertion fails.
    """

    connection = get_connection()

    if connection is None:
        return False

    cursor = None

    try:
        cursor = connection.cursor()

        query = """
            INSERT INTO detections (
                plate_text,
                track_id,
                confidence,
                timestamp,
                source
            )
            VALUES (%s, %s, %s, %s, %s)
        """

        values = (
            plate_text,
            track_id,
            confidence,
            timestamp,
            source
        )

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
    """
    Fetch all stored detections.

    Results are returned with the newest detection first.

    Returns:
        List of detection records.
    """

    connection = get_connection()

    if connection is None:
        return []

    cursor = None

    try:
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                id,
                plate_text,
                track_id,
                confidence,
                timestamp,
                source
            FROM detections
            ORDER BY timestamp DESC
        """

        cursor.execute(query)

        detections = cursor.fetchall()

        return detections

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
    """
    Fetch one detection by its database ID.

    Args:
        detection_id: ID of the detection.

    Returns:
        Detection dictionary if found.
        None if not found.
    """

    connection = get_connection()

    if connection is None:
        return None

    cursor = None

    try:
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                id,
                plate_text,
                track_id,
                confidence,
                timestamp,
                source
            FROM detections
            WHERE id = %s
        """

        cursor.execute(query, (detection_id,))

        detection = cursor.fetchone()

        return detection

    except Error as error:
        print(f"[DATABASE ERROR] {error}")
        return None

    finally:
        if cursor is not None:
            cursor.close()

        if connection.is_connected():
            connection.close()


# ============================================
# DATABASE TEST
# ============================================

if __name__ == "__main__":
    from datetime import datetime

    # Test MySQL connection
    connection = get_connection()

    if connection:
        print("[INFO] MySQL connection successful.")
        connection.close()
    else:
        print("[ERROR] MySQL connection failed.")

    # Test insertion
    success = save_detection(
        plate_text="MH12AB1234",
        track_id=1,
        confidence=0.95,
        timestamp=datetime.now(),
        source="test"
    )

    if success:
        print("[INFO] Test detection inserted successfully.")
    else:
        print("[ERROR] Test detection insert failed.")

    # Test fetching all detections
    detections = get_detections()

    print(f"[INFO] Total detections: {len(detections)}")

    for detection in detections:
        print(detection)

    # Test fetching one detection
    if detections:
        detection_id = detections[0]["id"]

        detection = get_detection_by_id(detection_id)

        print("[INFO] Single detection:")
        print(detection)
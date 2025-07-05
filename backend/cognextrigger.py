import logging
import mysql.connector
from CognexNativePy import NativeInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def connect_to_database(database: str) -> mysql.connector.MySQLConnection:
    """
    Establish a connection to the specified MySQL database.

    Args:
        database (str): The name of the database to connect to.

    Returns:
        mysql.connector.MySQLConnection: The MySQL connection object.
    """
    try:
        connection = mysql.connector.connect(
            host='136.16.182.41',
            port=3377,
            user='jcontramestre',
            password='904OBol6PY0mcIpN',
            database=database
        )
        logging.info(f"Connected to database: {database}")
        return connection
    except mysql.connector.Error as db_error:
        logging.error(f"Failed to connect to database {database}: {db_error}")
        raise

def get_variant_id_by_name(variant_name: str, database_name: str) -> int:
    """
    Get the idVariant based on the variantName from the Variants table.

    Args:
        variant_name (str): The name of the variant (Customer-Variant).
        database_name (str): The name of the MySQL database.

    Returns:
        int: The idVariant associated with the variant_name.
    """
    connection = None
    try:
        connection = connect_to_database(database_name)
        cursor = connection.cursor()

        cursor.execute('''SELECT idVariant FROM Variants WHERE variantName = %s''', (variant_name,))
        result = cursor.fetchone()

        if result:
            logging.info(f"Found idVariant for variant '{variant_name}': {result[0]}")
            return result[0]
        else:
            logging.warning(f"Variant '{variant_name}' not found in the database.")
            return None

    except mysql.connector.Error as db_error:
        logging.error(f"Database error while fetching idVariant: {db_error}")
        raise

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            logging.info("Database connection closed.")

def save_image_to_db(image_path: str, variant_name: str, database_name: str):
    """
    Save the pre-image to the ImagePreProcess table, associating it with the idVariant.

    Args:
        image_path (str): Path to the image file.
        variant_name (str): The name of the variant (Customer-Variant).
        database_name (str): The name of the MySQL database.
    """
    connection = None
    try:
        id_variant = get_variant_id_by_name(variant_name, database_name)

        if id_variant is None:
            logging.error("No valid idVariant found. Exiting function.")
            return

        connection = connect_to_database(database_name)
        cursor = connection.cursor()

        with open(image_path, 'rb') as img_file:
            image_blob = img_file.read()

        cursor.execute('''INSERT INTO ImagePreProcess (idVariant, preImage) VALUES (%s, %s)''', (id_variant, image_blob))
        connection.commit()
        logging.info("Pre-image saved to ImagePreProcess table successfully!")

    except mysql.connector.Error as db_error:
        logging.error(f"Database error while saving image: {db_error}")
        raise

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            logging.info("Database connection closed.")

def capture_and_save_image(variant_name: str, database_name: str, job_name: str, temp_image_path: str):
    """
    Capture an image from the Cognex vision system and save it to the database.

    Args:
        variant_name (str): The name of the variant (Customer-Variant).
        database_name (str): The name of the MySQL database.
        job_name (str): The job name to load on the Cognex system.
        temp_image_path (str): Temporary path to save the captured image.
    """
    try:
        native_interface = NativeInterface('192.168.0.253', 'admin', '')
        execution_and_online = native_interface.execution_and_online
        file_and_job = native_interface.file_and_job
        image = native_interface.image

        if file_and_job.get_file() != job_name:
            if execution_and_online.get_online() == 1:
                execution_and_online.set_online(0)
            file_and_job.load_file(job_name)

        if execution_and_online.get_online() == 0:
            execution_and_online.set_online(1)

        image_data = image.read_image()
        with open(temp_image_path, 'wb') as f:
            f.write(image_data["data"])

        logging.info(f"Image captured and saved to temporary path: {temp_image_path}")

        save_image_to_db(temp_image_path, variant_name, database_name)

        native_interface.close()
        logging.info("Cognex NativeInterface connection closed.")

    except Exception as e:
        logging.error(f"Error during image capture and save: {e}")
        raise

def main():
    variant_name = "Customer-Variant"  # Replace with the actual variant name
    database_name = "LiquidSealRobot"
    job_name = "1myJob.job"
    temp_image_path = "image.bmp"

    try:
        capture_and_save_image(variant_name, database_name, job_name, temp_image_path)
    except Exception as e:
        logging.critical(f"Critical failure in main: {e}")

if __name__ == '__main__':
    main()

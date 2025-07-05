import mysql.connector
import os
import glob
import argparse
from mysql.connector import connection

def connect_to_database(database: str) -> connection.MySQLConnection:
    """
    Establish a connection to the specified MySQL database.

    Args:
        database (str): Name of the database to connect to.

    Returns:
        connection.MySQLConnection: A MySQL connection object.
    """
    return mysql.connector.connect(
        host='192.168.10.42',
        port=3377,
        user='jcontramestre',
        password='904OBol6PY0mcIpN',
        database=database
    )

def get_compressor_details(serial_number: str) -> tuple:
    """
    Retrieve customer and customer variant details for a given compressor serial number.

    Args:
        serial_number (str): The serial number of the compressor.

    Returns:
        tuple: A tuple containing the customer and customer variant, or None if not found.
    """
    query = """
        SELECT Customer, Customer_Variant
        FROM Compressor_Build_Header
        WHERE Serial_Number = %s
    """
    with connect_to_database('CompressorDB') as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (serial_number,))
            result = cursor.fetchone()
    return result

def check_variant_existence(customer_variant: str) -> tuple:
    """
    Check if a customer variant exists in the GapfillerRobot.Variants table.

    Args:
        customer_variant (str): The name of the customer variant to check.

    Returns:
        tuple: A tuple containing the variant ID and variant program if found, or None if not found.
    """
    query = """
        SELECT idVariant, variantProgram
        FROM Variants
        WHERE variantName = %s
    """
    with connect_to_database('GapfillerRobot') as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (customer_variant,))
            result = cursor.fetchone()
    return result

def create_new_variant(customer_variant: str, variant_program: str) -> None:
    """
    Create a new customer variant in the GapfillerRobot.Variants table.

    Args:
        customer_variant (str): The name of the customer variant.
        variant_program (str): The program associated with the variant.

    Returns:
        None
    """
    query = """
        INSERT INTO Variants (variantName, variantProgram)
        VALUES (%s, %s)
    """
    with connect_to_database('GapfillerRobot') as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (customer_variant, variant_program))
            conn.commit()

def find_jcs_file(directory: str) -> str:
    """
    Find the first .jcs file in the given directory.

    Args:
        directory (str): The directory to search in.

    Returns:
        str: Path to the first .jcs file if found, otherwise None.
    """
    jcs_files = glob.glob(os.path.join(directory, '*.jcs'))
    return jcs_files[0] if jcs_files else None

def main(serial_number: str) -> str:
    """
    Main function to process compressor details and update variants.

    Args:
        serial_number (str): The serial number of the compressor.

    Returns:
        str: The customer variant or an error message if details are not found.
    """
    details = get_compressor_details(serial_number)
    if details:
        customer_variant = details
        
        # Check if the Customer Variant exists in the Variants table
        variant_details = check_variant_existence(customer_variant)
        
        # Locate the .jcs file if the variant does not already exist
        jcs_file = find_jcs_file('C:\\Users\\Model Shop\\Desktop')  # Update the path as necessary
        if jcs_file and not variant_details:
            with open(jcs_file, 'r') as file:
                variant_program = file.read()

            # Create a new variant only if it doesn't exist
            create_new_variant(customer_variant, variant_program)

        return customer_variant
    else:
        return "No details found for the given Serial Number."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process compressor details and update variants.')
    parser.add_argument('serial_number', type=str, help='Serial Number of the compressor')
    args = parser.parse_args()
    print(main(args.serial_number))

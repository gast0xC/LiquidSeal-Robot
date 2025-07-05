import mysql.connector
import os
import glob
import re
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
        host='136.16.182.41',
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
    Check if a customer variant exists in the LiquidSealRobot.Variants table.

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
    with connect_to_database('LiquidSealRobot') as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (customer_variant,))
            result = cursor.fetchone()
    return result

def create_new_variant(customer_variant: str, variant_program: str) -> None:
    """
    Create a new customer variant in the LiquidSealRobot.Variants table.

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
    with connect_to_database('LiquidSealRobot') as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (customer_variant, variant_program))
            conn.commit()

def find_and_process_txt_file(directory: str) -> str:
    """
    Find the first .txt file in the given directory, process its data, and return it as a string.

    Args:
        directory (str): The directory to search in.

    Returns:
        str: The processed data as a string.
    """
    # Define mappings for the first element transformation
    type_mapping = {
        'Start of Line Dispense': 'CP_S',
        'Line Passing': 'CP_P',
        'CP Arc Point': 'ARC',
        'End of Line Dispense': 'CP_E'
    }
    
    # Regex pattern to capture the first field and the rest of the values, allowing "-" in the last position
    pattern = r'^(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.-]+)$'

    # Search for the .txt file
    txt_files = glob.glob(os.path.join(directory, '*.txt'))
    if not txt_files:
        return None
    
    processed_data = []

    # Process the first .txt file found
    with open(txt_files[0], 'r') as infile:
        next(infile)  # Skip the first line
        
        for line in infile:
            match = re.match(pattern, line.strip())
            if match:
                # Extract parts from the regex groups
                label = match.group(1)
                x = match.group(2)
                y = match.group(3)
                z = match.group(4)
                parameter = match.group(5)
                
                # Map the label if it exists in type_mapping
                if label in type_mapping:
                    label = type_mapping[label]
                    # If it's 'CP_E' and parameter is '-', change it to '27.0'
                    if label == 'CP_E' and parameter == '-': #not even needed for robot_com.py
                        parameter = '27.0'
                    processed_data.append(f"{label},{x},{y},{z},{parameter}")
    
    # Add a header to the output
    processed_data.insert(0, "JR Points_CSV")

def main(serial_number: str) -> str:
    """
    Main function to process compressor details, update variants, and insert data from a .txt file.

    Args:
        serial_number (str): The serial number of the compressor.

    Returns:
        str: The customer variant or an error message if details are not found.
    """
    details = get_compressor_details(serial_number)
    #print(details)
    if details:
        customer_variant = details[1]
        
        # Check if the Customer Variant exists in the Variants table
        variant_details = check_variant_existence(customer_variant)
        #print(variant_details)
        # Locate and process the .txt file if the variant does not already exist
        if not variant_details:
            processed_data = find_and_process_txt_file('C:\\Users\\jcontram\\Desktop')  #'C:\\Users\\jcontram\\Desktop\\asdasdsadasd.txt
            
            if processed_data:
                # Insert the processed data into the Variants table
                create_new_variant(customer_variant, processed_data)

        return customer_variant
    else:
        return "No details found for the given Serial Number."

if __name__ == "__main__":
    
    #parser = argparse.ArgumentParser(description='Process compressor details and update variants.')
    #parser.add_argument('serial_number', type=str, help='Serial Number of the compressor')
    #args = parser.parse_args()
    #print(main(args.serial_number))
    main("G5CH453004G1756")
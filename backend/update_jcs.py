import mysql.connector
import os
import glob
import argparse
import re

def connect_to_database(database: str) -> mysql.connector.MySQLConnection:
    """
    Establish a connection to the specified MySQL database.

    Args:
        database (str): The name of the database to connect to.

    Returns:
        mysql.connector.MySQLConnection: The MySQL connection object.
    """
    return mysql.connector.connect(
        host='136.16.182.41',
        port=3377,
        user='jcontramestre',
        password='904OBol6PY0mcIpN',
        database=database
    )

def get_variant_id(customer_variant: str) -> int:
    """
    Retrieve the ID of a variant from the LiquidSealRobot database based on its name.

    Args:
        customer_variant (str): The name of the variant to look up.

    Returns:
        int: The ID of the variant if found, otherwise None.
    """
    query = "SELECT idVariant FROM Variants WHERE variantName = %s"
    with connect_to_database('LiquidSealRobot') as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (customer_variant,))
            result = cursor.fetchone()
    return result[0] if result else None

def update_variant_program(id_variant: int, new_variant_program: str) -> None:
    """
    Update the variant program in the database for a given variant ID.

    Args:
        id_variant (int): The ID of the variant to update.
        new_variant_program (str): The new variant program to set.
    """
    query = """
        UPDATE Variants
        SET variantProgram = %s
        WHERE idVariant = %s
    """
    with connect_to_database('LiquidSealRobot') as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (new_variant_program, id_variant))
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
                    if label == 'CP_E' and parameter == '-':  #not even needed for robot_com.py
                        parameter = '27.0'                     
                    processed_data.append(f"{label},{x},{y},{z},{parameter}")
    
    # Add a header to the output
    processed_data.insert(0, "JR Points_CSV")

    return "\n".join(processed_data)

def main(customer_variant: str) -> str:
    """
    Main function to update the variant program for a given customer variant.

    Args:
        customer_variant (str): The name of the customer variant to update.

    Returns:
        str: A message indicating the result of the operation.
    """
    id_variant = get_variant_id(customer_variant)
    
    if not id_variant:
        return f"No variant found with name {customer_variant}."

    jcs_data = find_and_process_txt_file('C:\\Users\\jcontram\\Desktop') 
    if not jcs_data:
        return "No .txt file found in the directory."

    # Save the processed data to a new .jcs file
    jcs_file_path = 'C:\\Users\\jcontram\\Desktop\\processed_data.jcs'
    with open(jcs_file_path, 'w') as jcs_file:
        jcs_file.write(jcs_data)

    # Open the newly saved .jcs file for reading
    with open(jcs_file_path, 'r') as file:
        variant_program = file.read()

    # Update the existing variant with the new variantProgram
    update_variant_program(id_variant, variant_program)
    
    return f"Variant {customer_variant} updated successfully."


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Update variant program for a given customer variant.')
    #parser.add_argument('customer_variant', type=str, help='Customer Variant to update')
    #args = parser.parse_args()
    #print(main(args.customer_variant))
    main("SPA2u_B")

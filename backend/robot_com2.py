import csv
import socket
import time
import mysql.connector
# IP address and port of the robot
robot_ip = "172.16.17.5"
robot_port = 10031  # Port for command communication

point_types = {
    "CP_S": "3030303030303232",  # CP Start Point
    "CP_P": "3030303030303234",  # CP Passing Point
    "ARC": "3030303030303136",   # CP Arc Point
    "CP_E": "3030303030303233"   # CP End Point
}

# Define a dictionary to map commands to their corresponding data lengths
command_data_lengths = {
    "4D31": "0000000C",
    "5231": "0000000C",
    "5230": "00000008",
    "5338": "0000000C",
    "5342": "0000000C",
    "5233": "0000000C",
    "5339": "0000000C",
    "5238": "0000000C",
    "5330": "00000014",
    "5430": "0000000C"
}

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

def get_variant_program(customer_variant: str) -> int:
    """
    Retrieve the variantProgram of a variant from the LiquidSealRobot database based on its name.

    Args:
        customer_variant (str): The name of the variant to look up.

    Returns:
        int: The ID of the variant if found, otherwise None.
    """
    query = "SELECT variantProgram FROM Variants WHERE variantName = %s"
    with connect_to_database('LiquidSealRobot') as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (customer_variant,))
            result = cursor.fetchone()
    return result[0] if result else None

def coordinate_to_byte_array(coordinate):
    # Step 1: Take the coordinate value
    # Step 2: Multiply it by 2000
    multiplied_value = int(coordinate * 2000)
    
    # Step 3: Convert the resulting number to hexadecimal
    hex_value = hex(multiplied_value)[2:].upper()
    
    # Step 4: Fill any leading zeros as needed (8 characters total)
    hex_value = hex_value.zfill(8)
    #print("Array of ASCII values:", hex_value)
    
    # Step 5: Convert the hexadecimal string to a list of hexadecimal representations
    hex_array = [hex(ord(char)) for char in hex_value]
    print("Coordinate Hexadecimal array:", hex_array)
    # Step 5: Convert the hexadecimal string to a string of hexadecimal representations
    hex_string = ''.join(hex(ord(char))[2:].upper() for char in hex_value)
    #print("Hexadecimal string:", hex_string)
    return hex_string

def linespeed_to_byte_array(linespeed):
    # Step 1: Take the coordinate value
    # Step 2: Multiply it by 10 (for each 0.1 mm)
    multiplied_value = int(linespeed * 10)
    
    # Step 3: Convert the resulting number to hexadecimal
    hex_value = hex(multiplied_value)[2:].upper()
    
    # Step 4: Fill any leading zeros as needed (8 characters total)
    hex_value = hex_value.zfill(8)
    #print("Array of ASCII values:", hex_value)
    
    # Step 5: Convert the hexadecimal string to a list of hexadecimal representations
    hex_array = [hex(ord(char)) for char in hex_value]
    print("Line speed Hexadecimal array:", hex_array)
    # Step 5: Convert the hexadecimal string to a string of hexadecimal representations
    hex_string = ''.join(hex(ord(char))[2:].upper() for char in hex_value)
    #print("Hexadecimal string:", hex_string)
    return hex_string

def send_data(s, data_frame):
    # Send the data frame
    s.sendall(bytes.fromhex(data_frame))  # Convert hexadecimal string to bytes and send
    print("Sending:", data_frame)

    # Extract the command from the data frame
    command = data_frame[12:16]
    print("Command:", command)

    # Determine the expected data length based on the command
    expected_length = int(command_data_lengths.get(command, "00000000"), 16)
    print("Expected Data Length:", expected_length)

    # Receive response
    response = b""  # Initialize response buffer
    while len(response) < expected_length:
        chunk = s.recv(expected_length - len(response))
        if not chunk:
            break  # Exit loop if no more data is received
        response += chunk
    response_length = len(response)
    print("Response Length:", response_length)
    print("Response (Hex):", response.hex())  # Print response in hexadecimal format

    # Parse and format the response
    identifier_index = response.find(b'\x03\x03')  # Find the index of the identifier
    if identifier_index != -1:
        data = response[identifier_index + 2:].hex()  # Extract data after the identifier
        print("Data (Hex):", data)
    else:
        print("Identifier not found in response")

def insert_point(s, point_type, x, y, z, line_speed):
    #Point Data Addition (SB)
    print("\tInserting Point")
    data_length = "0000008C"   # Data length 4 bytes      (140 bytes)
    identifier = "0303"        # Fixed identifier 2 bytes 
    program_number = "30303535" # 4 bytes
    #point_type = "3030303030303232" # 8 bytes
    point_position = "3030303030303031" + x + y + z + "303030303030303030303030303030303030303030303030" #56 bytes
    #line_speed = "3030303030303635" # 8 bytes
    zeros = "3030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030"   # 56 bytes
    point = point_type + point_position + line_speed + zeros
    command = "5342" + program_number + point  # Command data
    data_frame = data_length + identifier + command
    send_data(s, data_frame)

def init(s):
    #Mechanical Initialization (R0)
    print("\tMechanical Initialization")
    data_length = "00000008"   # Data length 4 bytes  
    identifier = "0303"        # Fixed identifier 2 bytes 
    command = "5230"           # Command data
    data_frame = data_length + identifier + command
    send_data(s, data_frame)
    print("\tWaiting for Mechanical Initialization...")
    time.sleep(10)
    
def program_creation(s):
    #Program Creation (S8)
    print("\tProgram Creation")
    data_length = "0000000C"   # Data length 4 bytes  
    identifier = "0303"        # Fixed identifier 2 bytes 
    program_number = "30303535"
    command = "5338" + program_number   # Command data
    data_frame = data_length + identifier + command
    send_data(s, data_frame)
    #time.sleep(1)

def program_change(s):
    #Change Program Number
    print("\tChange Program Number")
    data_length = "0000000C"   # Data length 4 bytes  
    identifier = "0303"        # Fixed identifier 2 bytes 
    program_number = "30303535"
    command = "5231" + program_number   # Command data

    data_frame = data_length + identifier + command
    send_data(s, data_frame)
    #time.sleep(1)

def program_start(s):
    #Start Designated Program Number(R8)
    print("\tStart Designated Program Number")
    data_length = "0000000C"   # Data length 4 bytes  
    identifier = "0303"        # Fixed identifier 2 bytes 
    program_number = "30303535"
    command = "5238" + program_number   # Command data

    data_frame = data_length + identifier + command
    send_data(s, data_frame)
    #time.sleep(1)

def program_deletion(s):
    #Program Deletion (S9)
    print("\tProgram Deletion")
    data_length = "0000000C"   # Data length 4 bytes  
    identifier = "0303"        # Fixed identifier 2 bytes   
    program_number = "30303535"
    command = "5339" + program_number   # Command data
    data_frame = data_length + identifier + command
    send_data(s, data_frame)

def read_variant_program(s, variant_program_text):
    print("\tReading variant program (from DB) and adding points")
    
    # Split the variant_program string into rows based on newline characters
    rows = variant_program_text.strip().split('\n')
    
    for row in rows:
        # Split the row into columns (assuming CSV format with commas separating columns)
        columns = row.split(',')
        
        # Process the row if it has exactly 5 elements
        if len(columns) == 5:
            for i, element in enumerate(columns):
                if i == 0:  # Processing the first element - index 0 (point type)
                    point_type = point_types.get(element, "Unknown")
                elif i == 1:  # Processing the 2nd element (x-coordinate)
                    x = coordinate_to_byte_array(float(element))
                elif i == 2:  # Processing the 3rd element (y-coordinate)
                    y = coordinate_to_byte_array(float(element))
                elif i == 3:  # Processing the 4th element (z-coordinate)
                    z = coordinate_to_byte_array(float(element))
                elif i == 4:  # Processing the last element - index 4 (line speed)
                    if element.isdigit():  # Check if it's a digit
                        line_speed = linespeed_to_byte_array(float(element))
                    else:
                        line_speed = linespeed_to_byte_array(10)  # Default value if not a digit
                    # Insert the point using the extracted data
                    insert_point(s, point_type, x, y, z, line_speed)
        else:
            print(f"Ignoring row: {row}. Expected 5 elements per row.")
        
def main(customer_variant: str) -> str:
    """
    Main function to update the variant program for a given customer variant.

    Args:
        customer_variant (str): The name of the customer variant to update.

    Returns:
        str: A message indicating the result of the operation.
    """
    variant_program = get_variant_program(customer_variant)
   # print(variant_program)
    # Create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # Connect to the robot
            s.connect((robot_ip, robot_port))
            program_deletion(s)
            init(s)
            program_creation(s)
            program_change(s)
            read_variant_program(s, variant_program)
            program_start(s)

        except ConnectionRefusedError:
            print("Connection refused. Make sure the robot is running and listening on the specified port.")

        except Exception as e:
            print("An error occurred:", e)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Update variant program for a given customer variant.')
    #parser.add_argument('customer_variant', type=str, help='Customer Variant to update')
    #args = parser.parse_args()
    #print(main(args.customer_variant))
    main("SPA2u_B")

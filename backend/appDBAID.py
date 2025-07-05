from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PySide6.QtGui import QStandardItem, QStandardItemModel, QColor, QPixmap
from PySide6.QtCore import Qt
import sys
import logging
import mysql.connector
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Customer Variant DB Aid")

        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        self.resize(int(screen_size.width() * 0.75), int(screen_size.height() * 0.75))  # 75% of screen size
        self.show()  # Show in normal mode

        # Database Configuration
        self.db_config = {
            'host': '136.16.182.41', #'192.168.10.42',    
            'port': 3377,
            'user': 'hprazere',
            'password': '3vq7mYtkRZnO85hB',
            'database': 'CompressorDB'
        }

        # List of Tables
        self.original_tables = [
            "CompressorDB.Customer",
            "CompressorDB.CMM_Report_Paths",
            "CompressorDB.Compressor_End_Item_Build_Reports",
            "CompressorDB.Compressor_End_Item_Checklist",
            "CompressorDB.Compressor_End_Item_Label_Config",
            "CompressorDB.Compressor_End_Item_Op_Mapping",
            "CompressorDB.Compressor_Weight_Limits",
            "CompressorDB.Inv_E_Test_Limits",
            "CompressorDB.Inv_Torque_Limits",
            "CompressorDB.Inverter_Active_Operations_Map",
            "CompressorDB.Inverter_Assembly_Gapfiller_Disp_Config",
            "CompressorDB.Inverter_Assembly_Liquid_Seal_Disp_Config",
            "CompressorDB.Inverter_Assembly_Op_Mapping",
            "CompressorDB.Inverter_Assembly_Torque_Config",
            "CompressorDB.Inverter_Board_Hipot_Config",
            "CompressorDB.Inverter_Board_Hipot_Limits",
            "CompressorDB.Inverter_Board_Software_Check",
            "CompressorDB.Inverter_CAL_Current_Config",
            "CompressorDB.Inverter_CAL_Voltage_Config",
            "CompressorDB.Inverter_Checklist_Maps",
            "CompressorDB.Inverter_Checks_Limits",
            "CompressorDB.Inverter_Discharge_Config",
            "CompressorDB.Inverter_Eletrical_Tests_Limits",
            "CompressorDB.Inverter_EQPotential_Config",
            "CompressorDB.Inverter_EQPotential_Tests_Limits",
            "CompressorDB.Inverter_HV_Config",
            "CompressorDB.Inverter_HW_SW",
            "CompressorDB.Inverter_PAR_Map",
            "CompressorDB.Inverter_Quiescent_Config",
            "CompressorDB.Inverter_Residual_Config",
            "CompressorDB.Inverter_Visual_Inspection_Config",
            "CompressorDB.MechPump_Assembly_Limits",
            "CompressorDB.MechPump_Assembly_Op_Mapping",
            "CompressorDB.MechPump_Leak_Config",
            "CompressorDB.MechPump_Main_Torque_Config",
            "CompressorDB.MechPump_Visual_Inspection_Config"
        ]

        # Main widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # Top Layout for dropdowns and buttons
        top_layout = QGridLayout()
        top_layout.setContentsMargins(5, 5, 5, 5)  # Reduce outer margins
        top_layout.setHorizontalSpacing(4)  # Tighten horizontal spacing
        top_layout.setVerticalSpacing(8)  # Adjust vertical spacing

        # Dropdown 1
        self.dropdown1_label = QLabel("Select a Customer Variant:")
        self.dropdown1_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Align label to the left
        self.dropdown1 = QComboBox()
        self.dropdown1.addItem("")  # Start with a blank item
        top_layout.addWidget(self.dropdown1_label, 0, 0, 1, 1, Qt.AlignLeft)  # Ensure left alignment
        top_layout.addWidget(self.dropdown1, 0, 1, 1, 2)  # Stretch dropdown to avoid misalignment

        # Dropdown 2
        self.dropdown2_label = QLabel("Select a Table:")
        self.dropdown2_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Align label to the left
        self.dropdown2 = QComboBox()
        self.dropdown2_model = QStandardItemModel()  # Initialize early
        self.dropdown2.setModel(self.dropdown2_model)
        top_layout.addWidget(self.dropdown2_label, 1, 0, 1, 1, Qt.AlignLeft)  # Ensure left alignment
        top_layout.addWidget(self.dropdown2, 1, 1, 1, 2)  # Stretch dropdown to align better

        # Buttons
        self.button1 = QPushButton("Update Record")
        self.button1.clicked.connect(self.update_record)
        self.button1.setMinimumSize(100, 40)  # Set button size
        self.button1.setStyleSheet("font-size: 14px;")
        top_layout.addWidget(self.button1, 0, 3)

        self.button2 = QPushButton("Create Record")
        self.button2.clicked.connect(self.create_record)
        self.button2.setMinimumSize(100, 40)  # Set button size
        self.button2.setStyleSheet("font-size: 14px;")
        top_layout.addWidget(self.button2, 1, 3)

        self.button3 = QPushButton("Exit")
        self.button3.clicked.connect(self.button3_clicked)
        self.button3.setMinimumSize(100, 40)  # Set button size
        self.button3.setStyleSheet("font-size: 14px;")
        top_layout.addWidget(self.button3, 2, 3)

        # Add top layout to main layout
        main_layout.addLayout(top_layout)


        # Table for displaying records
        self.record_table = QTableWidget()
        self.record_table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        self.record_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        main_layout.addWidget(self.record_table)

        # Add Image at the bottom
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        # Load image
        if getattr(sys, 'frozen', False):  # Checks if running as a bundled executable
            image_path = os.path.join(os.path.dirname(sys.executable), "awawa.jpeg")
        else:
            image_path = os.path.join(os.getcwd(), "awawa.jpeg")

        #print(image_path)
        pixmap = QPixmap(image_path)
        
        # Check if pixmap is loaded
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image_label.setText("Image could not be loaded")
        
        main_layout.addWidget(self.image_label)


        # Set main layout to central widget
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Populate Dropdown 1
        self.populate_dropdown1()

        # Connect signals AFTER initialization
        self.dropdown1.currentIndexChanged.connect(self.update_dropdown2)
        self.dropdown2.currentIndexChanged.connect(self.display_table_records)
        logging.debug("Initialization complete.")

    def populate_dropdown1(self):
        try:
            self.dropdown1.clear()  # Clear the dropdown before populating
            self.dropdown1.addItem("")  # Add the blank item at the beginning
            
            connection = mysql.connector.connect(**self.db_config)
            if connection.is_connected():
                cursor = connection.cursor()
                query = "SELECT DISTINCT Customer_Variant FROM ("
                for table in self.original_tables:
                    query += f"SELECT Customer_Variant FROM {table} UNION ALL "
                query = query[:-10] + ") AS all_variants"
                cursor.execute(query)
                variants = [row[0] for row in cursor.fetchall()]
                self.dropdown1.addItems(variants)  # Add all variants to the dropdown
                logging.debug(f"Dropdown1 populated with {len(variants)} variants.")
        except mysql.connector.Error as error:
            logging.error(f"Error retrieving customer variants: {error}")
        finally:
            if 'connection' in locals() and connection.is_connected():
                connection.close()


    def update_dropdown2(self):
        selected_variant = self.dropdown1.currentText()
        self.dropdown2_model.clear()
        try:
            connection = mysql.connector.connect(**self.db_config)
            if connection.is_connected():
                cursor = connection.cursor()
                for table in self.original_tables:
                    query = f"SELECT COUNT(*) FROM {table} WHERE Customer_Variant = %s"
                    cursor.execute(query, (selected_variant,))
                    count = cursor.fetchone()[0]
                    if count == 0:
                        self.add_colored_item(f"{table}", QColor("red"))
                    else:
                        self.add_colored_item(f"{table}", None)
        except mysql.connector.Error as error:
            logging.error(f"Error updating dropdown 2: {error}")
        finally:
            if 'connection' in locals() and connection.is_connected():
                connection.close()

    def add_colored_item(self, text, color):
        item = QStandardItem(text)
        if color:
            item.setForeground(color)
        self.dropdown2_model.appendRow(item)

    def display_table_records(self):
        selected_table = self.dropdown2.currentText()
        selected_variant = self.dropdown1.currentText()
        if not selected_table or not selected_variant:
            self.record_table.clear()
            return
        try:
            connection = mysql.connector.connect(**self.db_config)
            if connection.is_connected():
                cursor = connection.cursor()
                # Fetch records based on Customer_Variant
                query = f"SELECT * FROM {selected_table} WHERE Customer_Variant = %s"
                cursor.execute(query, (selected_variant,))
                records = cursor.fetchall()

                # Fetch column names
                column_query = f"SHOW COLUMNS FROM {selected_table}"
                cursor.execute(column_query)
                columns = [col[0] for col in cursor.fetchall()]

                # Update table widget
                self.record_table.setColumnCount(len(columns))
                self.record_table.setHorizontalHeaderLabels(columns)
                self.record_table.setRowCount(len(records))

                for row_idx, record in enumerate(records):
                    for col_idx, value in enumerate(record):
                        item = QTableWidgetItem(str(value))
                        self.record_table.setItem(row_idx, col_idx, item)

                self.record_table.resizeColumnsToContents()
                self.record_table.resizeRowsToContents()
        except mysql.connector.Error as error:
            logging.error(f"Error displaying table records: {error}")
        finally:
            if 'connection' in locals() and connection.is_connected():
                connection.close()

    def update_record(self):
        selected_table = self.dropdown2.currentText()
        if not selected_table:
            QMessageBox.warning(self, "Error", "No table selected for update.")
            logging.debug("No table selected for update. Exiting function.")
            return

        try:
            logging.debug(f"Attempting to update records in table: {selected_table}")
            connection = mysql.connector.connect(**self.db_config)
            if connection.is_connected():
                logging.debug("Database connection established successfully.")
                cursor = connection.cursor()
                rows_updated = 0

                # Iterate through each row in the GUI table
                for row in range(self.record_table.rowCount()):
                    logging.debug(f"Processing row {row + 1}/{self.record_table.rowCount()}.")

                    updates = []
                    update_values = []

                    # Always use the first column as the PK for the WHERE clause
                    primary_key_column = self.record_table.horizontalHeaderItem(0).text()
                    primary_key_item = self.record_table.item(row, 0)
                    primary_key_value = primary_key_item.text() if primary_key_item else None

                    if not primary_key_value:
                        logging.warning(f"Skipping row {row} due to missing primary key value in column '{primary_key_column}'.")
                        continue

                    # Prepare updates for all other columns
                    for col in range(1, self.record_table.columnCount()):  # Start from the second column
                        column_name = self.record_table.horizontalHeaderItem(col).text()
                        cell_item = self.record_table.item(row, col)
                        cell_value = cell_item.text() if cell_item else None

                        if cell_value is not None:
                            updates.append(f"{column_name} = %s")
                            update_values.append(cell_value)
                            logging.debug(f"Prepared update: {column_name} = {cell_value}")

                    # Append the primary key value at the end, matching the WHERE clause
                    where_clause = f"{primary_key_column} = %s"
                    update_values.append(primary_key_value)
                    logging.debug(f"Primary key for row {row}: {primary_key_column} = {primary_key_value}")

                    # Debugging for prepared updates
                    logging.debug(f"Row {row}: Updates to be made: {updates}")
                    logging.debug(f"Row {row}: Update values: {update_values}")

                    if updates:
                        update_query = f"UPDATE {selected_table} SET {', '.join(updates)} WHERE {where_clause}"
                        logging.debug(f"Constructed query for row {row}: {update_query}")
                        logging.debug(f"Executing query with values: {update_values}")
                        cursor.execute(update_query, update_values)
                        rows_updated += cursor.rowcount
                        logging.debug(f"Query executed successfully. Rows affected: {cursor.rowcount}")

                connection.commit()
                logging.info(f"{rows_updated} row(s) updated in table {selected_table}.")
                QMessageBox.information(self, "Success", f"{rows_updated} row(s) updated successfully.")
                logging.debug("Transaction committed successfully.")

                # Refresh the Customer Variant dropdown
                self.populate_dropdown1()
                logging.debug("Customer Variant dropdown refreshed.")
            else:
                logging.error("Failed to establish a database connection.")

        except mysql.connector.Error as error:
            logging.error(f"Error updating records: {error}")
            QMessageBox.critical(self, "Error", f"Error updating records: {error}")
            logging.debug(f"Exception occurred: {error}")

        finally:
            if 'connection' in locals() and connection.is_connected():
                connection.close()
                logging.debug("Database connection closed.")

    def create_record(self):
        selected_table = self.dropdown2.currentText()
        if not selected_table:
            QMessageBox.warning(self, "Error", "No table selected for record creation.")
            return

        try:
            logging.debug(f"Creating new records in table: {selected_table}")
            connection = mysql.connector.connect(**self.db_config)
            if connection.is_connected():
                cursor = connection.cursor()

                # Fetch column names
                column_names = [self.record_table.horizontalHeaderItem(col).text() for col in range(self.record_table.columnCount())]

                # Identify if the first column (primary key) is auto-incremented
                primary_key_column = column_names[0]
                cursor.execute(f"SHOW COLUMNS FROM {selected_table}")
                column_info = cursor.fetchall()
                is_auto_increment = any(col[0] == primary_key_column and 'auto_increment' in col for col in column_info)

                # Prepare insert query
                if is_auto_increment:
                    insert_query = f"INSERT INTO {selected_table} ({', '.join(column_names[1:])}) VALUES ({', '.join(['%s'] * (len(column_names) - 1))})"
                else:
                    insert_query = f"INSERT INTO {selected_table} ({', '.join(column_names)}) VALUES ({', '.join(['%s'] * len(column_names))})"

                logging.debug(f"Insert query prepared: {insert_query}")

                # Collect all rows of data
                rows_to_insert = []
                for row in range(self.record_table.rowCount()):
                    new_row = []
                    for col in range(self.record_table.columnCount()):
                        item = self.record_table.item(row, col)
                        value = item.text() if item else None
                        if not value and not is_auto_increment:  # Skip empty cells if auto-increment
                            QMessageBox.warning(self, "Error", f"Row {row + 1}, Column {column_names[col]} is empty. Fill all fields before creating a record.")
                            return
                        new_row.append(value)

                    if is_auto_increment:
                        # Exclude the primary key value for auto-incremented columns
                        new_row = new_row[1:]

                    rows_to_insert.append(tuple(new_row))

                # Execute insert queries
                for new_row in rows_to_insert:
                    logging.debug(f"Inserting row: {new_row}")
                    cursor.execute(insert_query, new_row)

                # Commit changes to the database
                connection.commit()
                logging.info(f"{len(rows_to_insert)} record(s) created successfully.")
                QMessageBox.information(self, "Success", f"{len(rows_to_insert)} record(s) created successfully.")

                # Refresh the dropdown with updated data
                self.populate_dropdown1()

        except mysql.connector.Error as error:
            logging.error(f"Error creating records: {error}")
            QMessageBox.critical(self, "Error", f"Error creating records: {error}")
        finally:
            if 'connection' in locals() and connection.is_connected():
                connection.close()
                logging.debug("Database connection closed.")


    def button3_clicked(self):
        logging.info("Exit button clicked!")
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2e2e2e;
        }
        QLabel {
            color: white;
            font-size: 14px;
        }
        QPushButton {
            background-color: #444444;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #555555;
        }
        QComboBox {
            background-color: #333333;
            color: white;
            border: 1px solid #444444;
            padding: 5px;
        }
        QComboBox QAbstractItemView {
            background-color: #333333;
            color: white;
            selection-background-color: #555555;
        }
        QTableWidget {
            background-color: #333333;
            color: white;
            gridline-color: #444444;
        }
        QTableWidget QHeaderView::section {
            background-color: #444444;
            color: white;
            border: 1px solid #555555;
        }
    """)

    window = SimpleApp()
    window.show()
    sys.exit(app.exec())

from google.cloud import bigtable
from google.cloud.bigtable import column_family
import yfinance as yf
import pandas as pd

# Configuración de Google Cloud y Bigtable
project_id = 'windy-tiger-405923'
instance_id = 'finance-predict-db'
table_id = 'data_yfinance'  # Replace with your actual table ID



def upload_data(data, column_family_id):
    # Inicializar el cliente de Bigtable
    client = bigtable.Client(project=project_id, admin=True)
    instance = client.instance(instance_id)

    # Crear una tabla con una familia de columnas
    table = instance.table(table_id)
    column_family = table.column_family(column_family_id)

    # Verificar si la tabla/columna existe y, si no, crearla
    if not table.exists():
        table.create()

    if column_family_id not in table.list_column_families():
        column_family.create()

    # upload data
    for index, row in data.iterrows():
        row_key = str(index)
        column_open = 'open'
        column_close = 'close'

        # Crear una fila y establecer un valor en la columna especificada
        bigtable_row = table.row(row_key)
        bigtable_row.set_cell(column_family_id, column_open, str(row['Open']))
        bigtable_row.set_cell(column_family_id, column_close, str(row['Close']))
        bigtable_row.commit()

def data_recolection(column_family_id, row_key):
    table = instance_id.table(table_id)

    # Leer datos de una fila específica
    cells = row.cells[column_family_id]
    row = table.read_row(row_key)
    
    data_variable = {}
    for cell in cells:
        timestamp_str = str(cell.timestamp)
        value = cell.value.decode('utf-8')  # Decodificar el valor de bytes si es necesario
        data_variable[timestamp_str] = value

    # Obtener celdas de la familia de columnas específica
    return row.cells[column_family_id]

def data_reference():
    client = bigtable.Client(project=project_id, admin=True)
    instance = client.instance(instance_id)

    # Obtener una referencia a la tabla
    table = instance.table(table_id)

    # Obtener las familias de columnas de la tabla
    column_families = table.list_column_families()

    # Imprimir las familias de columnas
    print("Familias de columnas de la tabla:")
    for column_family_id, column_family in column_families.items():
        print(f"ID de Familia: {column_family_id}")
        print(f"Configuración de Familia: {column_family}")
        print("------")
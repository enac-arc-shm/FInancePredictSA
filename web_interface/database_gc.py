from google.cloud import bigtable

# Set your Google Cloud Platform project ID and Bigtable instance ID
project_id = 'windy-tiger-405923'
instance_id = 'finance-predict-db'

# Set your Bigtable table ID and column family name
table_id = 'data_yfinance'  # Replace with your actual table ID
column_family_id = 'cf1'

# Construct the full table name
table_name = f'projects/{project_id}/instances/{instance_id}/tables/{table_id}'

# Initialize the Bigtable client
client = bigtable.Client(project=project_id, admin=True)
instance = client.instance(instance_id)

# Create a table with a single column family
table = instance.table(table_id)
column_family = table.column_family(column_family_id)

# Check if the table exists
if not table.exists():
    table.create()
    column_family.create()

# Insert data into the table
row_key = 'row_key_1'
column_id = 'column_id_1'
cell_value = b'Hello, Bigtable!'

row = table.row(row_key)
row.set_cell(column_family_id, column_id, cell_value)
row.commit()

print(f'Data inserted into Bigtable: RowKey={row_key}, ColumnID={column_id}, Value={cell_value.decode()}')

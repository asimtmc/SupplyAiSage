import sqlite3
import json

# Connect to the database
conn = sqlite3.connect('data/supply_chain.db')
cursor = conn.cursor()

# Query the model_parameter_cache table
cursor.execute("SELECT * FROM model_parameter_cache")
rows = cursor.fetchall()

print(f"Found {len(rows)} parameter records in the database.")

# Get column names
cursor.execute("PRAGMA table_info(model_parameter_cache)")
columns = cursor.fetchall()
column_names = [col[1] for col in columns]
print(f"Table columns: {column_names}")

for row in rows:
    row_dict = {column_names[i]: row[i] for i in range(len(row))}
    print(f"\nSKU: {row_dict['sku']}, Model: {row_dict['model_type']}")
    print(f"  ID: {row_dict['id']}")
    print(f"  Last Updated: {row_dict['last_updated']}")
    print(f"  Tuning Iterations: {row_dict['tuning_iterations']}")
    print(f"  Best Score: {row_dict['best_score']}")
    
    parameters = row_dict['parameters']
    if parameters is None:
        print("  Parameters: None (This is the issue!)")
    else:
        print(f"  Parameters (raw): {parameters}")
        try:
            params_dict = json.loads(parameters)
            if params_dict is None:
                print("  Parameters decoded to None")
            elif isinstance(params_dict, dict):
                if 'parameters' in params_dict and isinstance(params_dict['parameters'], dict):
                    print("  Parameters (nested):")
                    for key, value in params_dict['parameters'].items():
                        print(f"    {key}: {value}")
                else:
                    print("  Parameters (flat):")
                    for key, value in params_dict.items():
                        print(f"    {key}: {value}")
            else:
                print(f"  Parameters has unexpected type: {type(params_dict)}")
        except json.JSONDecodeError:
            print(f"  Error decoding parameters: {parameters}")
        except Exception as e:
            print(f"  Error processing parameters: {str(e)}")

# Print all records in the database for reference
print("\nAll model_parameter_cache records:")
cursor.execute("SELECT id, sku, model_type, parameters, best_score FROM model_parameter_cache")
for row in cursor.fetchall():
    print(f"ID: {row[0]}, SKU: {row[1]}, Model: {row[2]}, Best Score: {row[4]}, Params Length: {len(str(row[3])) if row[3] else 'None'}")

# Close the connection
conn.close()
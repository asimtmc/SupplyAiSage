import sqlite3
import json

# Connect to the database
conn = sqlite3.connect('data/supply_chain.db')
cursor = conn.cursor()

# Query the model_parameter_cache table
cursor.execute("SELECT sku, model_type, parameters FROM model_parameter_cache LIMIT 5")
rows = cursor.fetchall()

# Print each row's parameters
for row in rows:
    sku = row[0]
    model_type = row[1]
    params_str = row[2]
    print(f"\nSKU: {sku}, Model: {model_type}")
    
    try:
        # Try to parse the parameters as JSON
        if params_str is not None:
            params_json = json.loads(params_str)
            print("Parameters (parsed):")
            print(json.dumps(params_json, indent=2))
        else:
            print("Parameters: None")
    except json.JSONDecodeError:
        print(f"Could not parse parameters as JSON: {params_str}")

# Close the connection
conn.close()
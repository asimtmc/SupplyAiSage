import sqlite3
import json
import uuid
from datetime import datetime, timedelta
import pandas as pd

def save_model_parameters(sku, model_type, parameters, metrics=None):
    """Save model parameters to the database.
    
    Args:
        sku (str): The SKU identifier
        model_type (str): The type of model ('auto_arima', 'prophet', etc.)
        parameters (dict): Dictionary of parameters for the model
        metrics (dict, optional): Dictionary of evaluation metrics
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect('data/supply_chain.db')
        cursor = conn.cursor()
        
        # Check if parameters exist for this SKU and model
        cursor.execute(
            "SELECT id FROM model_parameter_cache WHERE sku = ? AND model_type = ?",
            (sku, model_type)
        )
        existing = cursor.fetchone()
        
        # Convert parameters to JSON
        params_json = json.dumps(parameters)
        metrics_json = json.dumps(metrics) if metrics else None
        
        timestamp = datetime.now()
        
        if existing:
            # Update existing record
            cursor.execute(
                """UPDATE model_parameter_cache 
                   SET parameters = ?, last_updated = ?, metrics = ?
                   WHERE sku = ? AND model_type = ?""",
                (params_json, timestamp, metrics_json, sku, model_type)
            )
        else:
            # Create new record
            cursor.execute(
                """INSERT INTO model_parameter_cache 
                   (id, sku, model_type, parameters, last_updated, metrics)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), sku, model_type, params_json, timestamp, metrics_json)
            )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving parameters: {str(e)}")
        return False

def get_model_parameters(sku, model_type):
    """Get model parameters from the database.
    
    Args:
        sku (str): The SKU identifier
        model_type (str): The type of model ('auto_arima', 'prophet', etc.)
    
    Returns:
        dict: Dictionary of parameters for the model, or None if not found
    """
    try:
        conn = sqlite3.connect('data/supply_chain.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            """SELECT parameters, last_updated, metrics 
               FROM model_parameter_cache 
               WHERE sku = ? AND model_type = ?""",
            (sku, model_type)
        )
        result = cursor.fetchone()
        
        if result:
            parameters = json.loads(result['parameters'])
            
            # Add last updated timestamp
            parameters['last_updated'] = datetime.fromisoformat(result['last_updated'])
            
            # Add metrics if available
            if result['metrics']:
                parameters['metrics'] = json.loads(result['metrics'])
            
            conn.close()
            return parameters
        else:
            conn.close()
            return None
    except Exception as e:
        print(f"Error retrieving parameters: {str(e)}")
        return None

def get_parameters_update_required(sku, model_type, days_threshold=7):
    """Check if parameters need to be updated based on last update time.
    
    Args:
        sku (str): The SKU identifier
        model_type (str): The type of model
        days_threshold (int): Number of days before parameters should be updated
        
    Returns:
        bool: True if update is needed, False otherwise
    """
    try:
        conn = sqlite3.connect('data/supply_chain.db')
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT last_updated FROM model_parameter_cache WHERE sku = ? AND model_type = ?",
            (sku, model_type)
        )
        result = cursor.fetchone()
        
        if result:
            last_updated = datetime.fromisoformat(result[0])
            threshold_date = datetime.now() - timedelta(days=days_threshold)
            
            if last_updated < threshold_date:
                conn.close()
                return True
            else:
                conn.close()
                return False
        else:
            # No parameters found, so update is required
            conn.close()
            return True
    except Exception as e:
        print(f"Error checking parameter update: {str(e)}")
        return True

def save_forecast_result(sku, model, forecast_data, metadata=None):
    """Save forecast results to the database.
    
    Args:
        sku (str): The SKU identifier
        model (str): The model name
        forecast_data (dict): Dictionary containing forecast results
        metadata (dict, optional): Additional metadata about the forecast
        
    Returns:
        str: The ID of the saved forecast, or None if there was an error
    """
    try:
        conn = sqlite3.connect('data/supply_chain.db')
        cursor = conn.cursor()
        
        forecast_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Convert forecast data to JSON
        forecast_json = json.dumps(forecast_data)
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute(
            """INSERT INTO forecast_results 
               (id, sku, model, forecast_date, forecast_data, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (forecast_id, sku, model, timestamp, forecast_json, metadata_json)
        )
        
        conn.commit()
        conn.close()
        return forecast_id
    except Exception as e:
        print(f"Error saving forecast: {str(e)}")
        return None

class ModelParameterCache:
    """Class to handle model parameter caching operations"""
    
    def __init__(self, db_path='data/supply_chain.db'):
        """Initialize with database path"""
        self.db_path = db_path
    
    def get_all_parameters(self):
        """Get all cached parameters as a DataFrame"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT id, sku, model_type, parameters, last_updated, metrics FROM model_parameter_cache"
            
            # Load into dataframe
            df = pd.read_sql_query(query, conn)
            
            # Parse JSON columns
            if not df.empty:
                df['parameters'] = df['parameters'].apply(json.loads)
                df['metrics'] = df['metrics'].apply(lambda x: json.loads(x) if x else None)
            
            conn.close()
            return df
        except Exception as e:
            print(f"Error getting parameters: {str(e)}")
            return pd.DataFrame()

# Create a session factory
class SessionFactory:
    """Factory for creating database connections"""
    
    @staticmethod
    def get_connection():
        """Get a new database connection"""
        return sqlite3.connect('data/supply_chain.db')

import os
import io
import uuid
import base64
import pandas as pd
from sqlalchemy import create_engine, Column, String, LargeBinary, DateTime, Text, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from sqlalchemy.sql import text

# Create database directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Use SQLite database as the PostgreSQL endpoint is disabled
# In a production environment, you would want to use the PostgreSQL database when available
DATABASE_URL = "sqlite:///data/supply_chain.db"
print(f"Using SQLite database: {DATABASE_URL}")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True
)
Base = declarative_base()

# Define the tables
class UploadedFile(Base):
    __tablename__ = 'uploaded_files'

    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    file_data = Column(LargeBinary, nullable=False)

class ForecastResult(Base):
    __tablename__ = 'forecast_results'

    id = Column(String(36), primary_key=True)
    sku = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    forecast_date = Column(DateTime, default=datetime.now)
    forecast_periods = Column(Integer, nullable=False)
    mape = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    forecast_data = Column(Text, nullable=False)  # JSON string of forecast values
    model_params = Column(Text, nullable=True)    # JSON string of model parameters

class ModelParameterCache(Base):
    __tablename__ = 'model_parameter_cache'
    
    id = Column(String(36), primary_key=True)
    sku = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    parameters = Column(Text, nullable=False)  # JSON string of optimized parameters
    last_updated = Column(DateTime, default=datetime.now)
    tuning_iterations = Column(Integer, default=0)  # Number of tuning iterations performed
    best_score = Column(Float, nullable=True)  # Best score achieved during tuning

class SecondarySales(Base):
    __tablename__ = 'secondary_sales'
    
    id = Column(String(36), primary_key=True)
    sku = Column(String(100), nullable=False)
    date = Column(DateTime, nullable=False)
    primary_sales = Column(Float, nullable=False)
    estimated_secondary_sales = Column(Float, nullable=False)
    noise = Column(Float, nullable=False)  # Difference between primary and secondary
    created_at = Column(DateTime, default=datetime.now)
    algorithm_used = Column(String(100), nullable=False)  # Which algorithm was used to calculate
    
# Create the tables
Base.metadata.create_all(engine)

# Create a session factory
SessionFactory = sessionmaker(bind=engine)

def save_uploaded_file(file, file_type, description=None):
    """
    Save an uploaded file to the database
    If a file of the same type already exists, delete it first

    Parameters:
    -----------
    file : UploadedFile object
        The file object from Streamlit's file_uploader
    file_type : str
        Type of file (e.g., 'sales_data', 'bom_data', 'supplier_data')
    description : str, optional
        Optional description of the file

    Returns:
    --------
    str
        ID of the saved file
    """
    session = None
    try:
        # Make sure the data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Create database session
        session = SessionFactory()

        # Check if file of this type already exists
        print(f"Checking for existing files of type: {file_type}")
        existing_files = session.query(UploadedFile).filter(
            UploadedFile.file_type == file_type
        ).all()

        # Delete existing files if any
        if existing_files:
            print(f"Found {len(existing_files)} existing files to delete")
            for existing_file in existing_files:
                session.delete(existing_file)

            # Use confirm_deleted_rows=False to suppress the warning
            session.commit()
            print(f"Deleted existing files of type: {file_type}")

        # Generate a unique ID
        file_id = str(uuid.uuid4())

        # Read the file data
        file.seek(0)  # Reset file pointer to beginning
        file_data = file.read()
        print(f"Read {len(file_data)} bytes from file: {file.name}")

        # Create a new UploadedFile record
        new_file = UploadedFile(
            id=file_id,
            filename=file.name,
            file_type=file_type,
            description=description,
            file_data=file_data
        )

        # Add to session and commit
        session.add(new_file)
        session.commit()
        print(f"Successfully saved file {file.name} with ID: {file_id}")

        return file_id
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        import traceback
        traceback.print_exc()
        if session:
            session.rollback()
        return None
    finally:
        if session:
            session.close()

def get_all_files():
    """
    Get all files from the database

    Returns:
    --------
    list
        List of files with id, filename, file_type, description, created_at
    """
    try:
        session = SessionFactory()
        files = session.query(
            UploadedFile.id, 
            UploadedFile.filename, 
            UploadedFile.file_type, 
            UploadedFile.description, 
            UploadedFile.created_at
        ).all()

        # Convert to list of dictionaries
        files_list = []
        for file in files:
            files_list.append({
                'id': file.id,
                'filename': file.filename,
                'file_type': file.file_type,
                'description': file.description,
                'created_at': file.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })

        return files_list
    except Exception as e:
        raise e
    finally:
        if session:
            session.close()

def get_file_by_id(file_id):
    """
    Get a file by its ID

    Parameters:
    -----------
    file_id : str
        ID of the file to retrieve

    Returns:
    --------
    tuple
        (filename, file_data_bytes)
    """
    try:
        session = SessionFactory()
        file = session.query(UploadedFile).filter(UploadedFile.id == file_id).first()

        if file:
            return (file.filename, file.file_data)
        else:
            return None
    except Exception as e:
        raise e
    finally:
        if session:
            session.close()

def get_file_by_type(file_type):
    """
    Get the most recent file of a specific type

    Parameters:
    -----------
    file_type : str
        Type of file to retrieve (e.g., 'sales_data', 'bom_data', 'supplier_data')

    Returns:
    --------
    tuple
        (filename, pandas_dataframe) or None if no file found
    """
    session = None
    try:
        # Create session and check for the file
        # No need to check SQLite file existence when using PostgreSQL
        session = SessionFactory()
        
        # Debug - check if the table exists
        try:
            file_count = session.query(UploadedFile).count()
            print(f"Total files in database: {file_count}")
        except Exception as table_err:
            print(f"Error checking UploadedFile table: {str(table_err)}")
            
        # Query the specific file type
        file = session.query(UploadedFile).filter(
            UploadedFile.file_type == file_type
        ).order_by(
            UploadedFile.created_at.desc()
        ).first()

        if file:
            print(f"Found file: {file.filename} (ID: {file.id}, Size: {len(file.file_data) if file.file_data else 0} bytes)")
            if not file.file_data or len(file.file_data) == 0:
                print(f"Error: File data is empty for {file.filename}")
                return None
                
            try:
                # Convert bytes data to pandas DataFrame
                file_bytes = io.BytesIO(file.file_data)
                df = pd.read_excel(file_bytes)
                print(f"Successfully parsed file {file.filename} - found {len(df)} rows")
                return (file.filename, df)
            except Exception as e:
                print(f"Error parsing file data for {file.filename}: {str(e)}")
                return None
        else:
            print(f"No file of type '{file_type}' found in database")
            return None
    except Exception as e:
        print(f"Database error in get_file_by_type: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if session:
            session.close()

def file_type_exists(file_type):
    """
    Check if a file of a specific type already exists in the database

    Parameters:
    -----------
    file_type : str
        Type of file to check (e.g., 'sales_data', 'bom_data', 'supplier_data')

    Returns:
    --------
    bool
        True if file exists, False otherwise
    """
    try:
        session = SessionFactory()
        file_exists = session.query(UploadedFile).filter(
            UploadedFile.file_type == file_type
        ).first() is not None

        return file_exists
    except Exception as e:
        raise e
    finally:
        if session:
            session.close()

def delete_file(file_id):
    """
    Delete a file by its ID

    Parameters:
    -----------
    file_id : str
        ID of the file to delete

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        session = SessionFactory()
        file = session.query(UploadedFile).filter(UploadedFile.id == file_id).first()

        if file:
            session.delete(file)
            session.commit()
            return True
        else:
            return False
    except Exception as e:
        if session:
            session.rollback()
        raise e
    finally:
        if session:
            session.close()

def save_forecast_result(sku, model_type, forecast_periods, mape, rmse, mae, forecast_data, model_params):
    """
    Save forecast results to the database

    Parameters:
    -----------
    sku : str
        SKU identifier
    model_type : str
        Type of forecasting model used (e.g., 'arima', 'prophet')
    forecast_periods : int
        Number of periods forecasted
    mape : float
        Mean Absolute Percentage Error
    rmse : float
        Root Mean Squared Error
    mae : float
        Mean Absolute Error
    forecast_data : str
        JSON string containing forecast data
    model_params : str
        JSON string containing model parameters

    Returns:
    --------
    bool
        True if save successful, False otherwise
    """
    try:
        session = SessionFactory()

        # Generate a unique ID for this forecast
        forecast_id = str(uuid.uuid4())

        # Convert numpy types to Python native types
        import numpy as np
        if isinstance(mape, (np.float64, np.float32, np.int64, np.int32)):
            mape = float(mape)
        if isinstance(rmse, (np.float64, np.float32, np.int64, np.int32)):
            rmse = float(rmse)
        if isinstance(mae, (np.float64, np.float32, np.int64, np.int32)):
            mae = float(mae)

        # Create a new forecast record
        new_forecast = ForecastResult(
            id=forecast_id,
            sku=sku,
            model_type=model_type,
            forecast_date=datetime.now(),
            forecast_periods=forecast_periods,
            mape=mape,
            rmse=rmse,
            mae=mae,
            forecast_data=forecast_data,
            model_params=model_params
        )

        # Add to session and commit
        session.add(new_forecast)
        session.commit()

        return True

    except Exception as e:
        print(f"Error saving forecast to database: {str(e)}")
        if session:
            session.rollback()
        return False


def get_forecast_history(sku=None):
    """
    Get forecast history from the database

    Parameters:
    -----------
    sku : str, optional
        SKU identifier to filter results. If None, returns all forecasts.

    Returns:
    --------
    list
        List of forecast results with id, sku, model_type, forecast_date, mape, rmse
    """
    try:
        session = SessionFactory()

        query = session.query(
            ForecastResult.id,
            ForecastResult.sku,
            ForecastResult.model_type,
            ForecastResult.forecast_date,
            ForecastResult.mape,
            ForecastResult.rmse,
            ForecastResult.mae
        )

        if sku:
            query = query.filter(ForecastResult.sku == sku)

        forecasts = query.order_by(ForecastResult.forecast_date.desc()).all()

        # Convert to list of dictionaries
        forecasts_list = []
        for forecast in forecasts:
            forecasts_list.append({
                'id': forecast.id,
                'sku': forecast.sku,
                'model_type': forecast.model_type,
                'forecast_date': forecast.forecast_date.strftime('%Y-%m-%d %H:%M:%S'),
                'mape': forecast.mape,
                'rmse': forecast.rmse,
                'mae': forecast.mae
            })

        return forecasts_list
    except Exception as e:
        raise e
    finally:
        if session:
            session.close()

def get_forecast_details(forecast_id):
    """
    Get detailed forecast information by its ID

    Parameters:
    -----------
    forecast_id : str
        ID of the forecast to retrieve

    Returns:
    --------
    dict
        Forecast details including forecast_data and model_params
    """
    try:
        session = SessionFactory()
        forecast = session.query(ForecastResult).filter(ForecastResult.id == forecast_id).first()

        if forecast:
            return {
                'id': forecast.id,
                'sku': forecast.sku,
                'model_type': forecast.model_type,
                'forecast_date': forecast.forecast_date.strftime('%Y-%m-%d %H:%M:%S'),
                'forecast_periods': forecast.forecast_periods,
                'mape': forecast.mape,
                'rmse': forecast.rmse,
                'mae': forecast.mae,
                'forecast_data': forecast.forecast_data,
                'model_params': forecast.model_params
            }
        else:
            return None
    except Exception as e:
        raise e
    finally:
        if session:
            session.close()

def save_model_parameters(sku, model_type, parameters, best_score=None, tuning_iterations=1):
    """
    Save optimized model parameters to the cache

    Parameters:
    -----------
    sku : str
        SKU identifier
    model_type : str
        Type of forecasting model (e.g., 'arima', 'prophet', 'xgboost')
    parameters : dict
        Dictionary of optimized parameters
    best_score : float, optional
        Best score achieved during tuning (lower is better)
    tuning_iterations : int, optional
        Number of tuning iterations performed

    Returns:
    --------
    bool
        True if save successful, False otherwise
    """
    try:
        session = SessionFactory()
        
        # Check if entry already exists
        existing = session.query(ModelParameterCache).filter(
            ModelParameterCache.sku == sku,
            ModelParameterCache.model_type == model_type
        ).first()
        
        # Convert parameters to JSON string
        import json
        if not isinstance(parameters, str):
            parameters_json = json.dumps(parameters)
        else:
            parameters_json = parameters
        
        if existing:
            # Update existing entry
            existing.parameters = parameters_json
            existing.last_updated = datetime.now()
            existing.tuning_iterations += tuning_iterations
            
            # Only update best_score if it's better (lower) than the existing one
            if best_score is not None:
                if existing.best_score is None or best_score < existing.best_score:
                    existing.best_score = best_score
        else:
            # Create new entry
            cache_id = str(uuid.uuid4())
            new_cache = ModelParameterCache(
                id=cache_id,
                sku=sku,
                model_type=model_type,
                parameters=parameters_json,
                last_updated=datetime.now(),
                tuning_iterations=tuning_iterations,
                best_score=best_score
            )
            session.add(new_cache)
        
        session.commit()
        return True
    
    except Exception as e:
        print(f"Error saving model parameters: {str(e)}")
        if session:
            session.rollback()
        return False
    finally:
        if session:
            session.close()

def get_model_parameters(sku, model_type):
    """
    Get cached model parameters

    Parameters:
    -----------
    sku : str
        SKU identifier
    model_type : str
        Type of forecasting model (e.g., 'arima', 'prophet', 'xgboost')

    Returns:
    --------
    dict
        Dictionary of optimized parameters or None if not found
    """
    try:
        session = SessionFactory()
        cache = session.query(ModelParameterCache).filter(
            ModelParameterCache.sku == sku,
            ModelParameterCache.model_type == model_type
        ).first()
        
        if cache:
            import json
            try:
                # Try to parse parameters as JSON
                parameters = json.loads(cache.parameters)
                return {
                    'parameters': parameters,
                    'last_updated': cache.last_updated,
                    'tuning_iterations': cache.tuning_iterations,
                    'best_score': cache.best_score
                }
            except json.JSONDecodeError:
                print(f"Error parsing JSON parameters for {sku}, {model_type}")
                return None
        
        return None
    
    except Exception as e:
        print(f"Error retrieving model parameters: {str(e)}")
        return None
    finally:
        if session:
            session.close()

def get_all_model_parameters():
    """
    Get all model parameters in the database

    Returns:
    --------
    dict
        Dictionary mapping SKU and model types to their parameters
    """
    try:
        session = SessionFactory()
        all_params = session.query(ModelParameterCache).all()
        
        result = {}
        for param in all_params:
            if param.sku not in result:
                result[param.sku] = {}
                
            import json
            try:
                parameters = json.loads(param.parameters)
                result[param.sku][param.model_type] = {
                    'parameters': parameters,
                    'last_updated': param.last_updated,
                    'tuning_iterations': param.tuning_iterations,
                    'best_score': param.best_score
                }
            except json.JSONDecodeError:
                print(f"Error parsing JSON parameters for {param.sku}, {param.model_type}")
                
        return result
    
    except Exception as e:
        print(f"Error retrieving all model parameters: {str(e)}")
        return {}
    finally:
        if session:
            session.close()

def get_flat_model_parameters():
    """
    Get all model parameters in a flat format suitable for tabular display
    
    Returns:
    --------
    list of dict
        List of dictionaries with keys: sku_code, sku_name, model_name, parameter_name, parameter_value
    """
    try:
        session = SessionFactory()
        all_params = session.query(ModelParameterCache).all()
        
        flat_results = []
        import json
        
        for param in all_params:
            try:
                parameters = json.loads(param.parameters)
                
                # For each parameter, create a separate row
                for param_name, param_value in parameters.items():
                    # Convert value to string for consistent display
                    if isinstance(param_value, (list, dict)):
                        value_str = json.dumps(param_value)
                    else:
                        value_str = str(param_value)
                    
                    flat_results.append({
                        'sku_code': param.sku,
                        'sku_name': param.sku,  # Using SKU code as name since we don't have separate names
                        'model_name': param.model_type.upper(),
                        'parameter_name': param_name,
                        'parameter_value': value_str
                    })
            except json.JSONDecodeError:
                print(f"Error parsing JSON parameters for {param.sku}, {param.model_type}")
        
        return flat_results
    except Exception as e:
        print(f"Error retrieving flat model parameters: {str(e)}")
        return []
    finally:
        if session:
            session.close()

def get_parameters_update_required(sku, model_type, days_threshold=7):
    """
    Check if parameters need updating based on age
    
    Parameters:
    -----------
    sku : str
        SKU identifier
    model_type : str
        Type of forecasting model
    days_threshold : int, optional
        Number of days after which parameters are considered stale
        
    Returns:
    --------
    bool
        True if parameters need updating, False otherwise
    """
    try:
        session = SessionFactory()
        cache = session.query(ModelParameterCache).filter(
            ModelParameterCache.sku == sku,
            ModelParameterCache.model_type == model_type
        ).first()
        
        if not cache:
            return True  # No cached parameters, update required
        
        # Check if parameters are older than threshold
        age = datetime.now() - cache.last_updated
        return age.days >= days_threshold
    
    except Exception as e:
        print(f"Error checking parameters age: {str(e)}")
        return True  # Default to updating if error occurs
    finally:
        if session:
            session.close()

def delete_old_parameters(days_threshold=90):
    """
    Delete parameter cache entries older than threshold
    
    Parameters:
    -----------
    days_threshold : int, optional
        Number of days after which parameters are deleted
        
    Returns:
    --------
    int
        Number of entries deleted
    """
    try:
        session = SessionFactory()
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        # Find and delete old entries
        old_entries = session.query(ModelParameterCache).filter(
            ModelParameterCache.last_updated < cutoff_date
        ).all()
        
        count = len(old_entries)
        for entry in old_entries:
            session.delete(entry)
        
        session.commit()
        return count
    
    except Exception as e:
        print(f"Error deleting old parameters: {str(e)}")
        if session:
            session.rollback()
        return 0
    finally:
        if session:
            session.close()

def save_secondary_sales(sku, date, primary_sales, estimated_secondary_sales, noise, algorithm_used):
    """
    Save secondary sales calculation results to the database
    
    Parameters:
    -----------
    sku : str
        SKU identifier
    date : datetime
        Date of the sales data
    primary_sales : float
        Primary sales value from the uploaded data
    estimated_secondary_sales : float
        Calculated secondary sales value
    noise : float
        Calculated noise (difference between primary and secondary)
    algorithm_used : str
        Name of algorithm used for calculation
        
    Returns:
    --------
    str
        ID of the saved record
    """
    try:
        session = SessionFactory()
        
        # Check if record already exists for this SKU and date
        existing = session.query(SecondarySales).filter(
            SecondarySales.sku == sku,
            SecondarySales.date == date
        ).first()
        
        if existing:
            # Update existing record
            existing.primary_sales = primary_sales
            existing.estimated_secondary_sales = estimated_secondary_sales
            existing.noise = noise
            existing.algorithm_used = algorithm_used
            existing.created_at = datetime.now()
            record_id = existing.id
        else:
            # Generate a unique ID
            record_id = str(uuid.uuid4())
            
            # Create new record
            new_record = SecondarySales(
                id=record_id,
                sku=sku,
                date=date,
                primary_sales=primary_sales,
                estimated_secondary_sales=estimated_secondary_sales,
                noise=noise,
                algorithm_used=algorithm_used
            )
            session.add(new_record)
        
        session.commit()
        return record_id
    
    except Exception as e:
        if session:
            session.rollback()
        print(f"Error saving secondary sales: {str(e)}")
        raise e
    finally:
        if session:
            session.close()

def get_secondary_sales(sku=None):
    """
    Get secondary sales data from the database
    
    Parameters:
    -----------
    sku : str, optional
        SKU identifier to filter results. If None, returns data for all SKUs.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with secondary sales data
    """
    try:
        session = SessionFactory()
        
        query = session.query(
            SecondarySales.sku,
            SecondarySales.date,
            SecondarySales.primary_sales,
            SecondarySales.estimated_secondary_sales,
            SecondarySales.noise,
            SecondarySales.algorithm_used
        )
        
        if sku:
            query = query.filter(SecondarySales.sku == sku)
            
        results = query.order_by(SecondarySales.sku, SecondarySales.date).all()
        
        # Convert to DataFrame
        data = []
        for record in results:
            data.append({
                'sku': record.sku,
                'date': record.date,
                'primary_sales': record.primary_sales,
                'secondary_sales': record.estimated_secondary_sales,
                'noise': record.noise,
                'algorithm': record.algorithm_used
            })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error retrieving secondary sales: {str(e)}")
        return pd.DataFrame()
    finally:
        if session:
            session.close()

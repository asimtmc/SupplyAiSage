import os
import io
import uuid
import base64
import pandas as pd
from sqlalchemy import create_engine, Column, String, LargeBinary, DateTime, Text, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from sqlalchemy.sql import text

# Get the database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

# Create the engine
engine = create_engine(DATABASE_URL)
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
    try:
        session = SessionFactory()

        # Check if file of this type already exists
        existing_files = session.query(UploadedFile).filter(
            UploadedFile.file_type == file_type
        ).all()

        # Delete existing files if any
        if existing_files:
            for existing_file in existing_files:
                session.delete(existing_file)

            # Use confirm_deleted_rows=False to suppress the warning
            session.commit()

        # Generate a unique ID
        file_id = str(uuid.uuid4())

        # Read the file data
        file_data = file.read()

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

        return file_id
    except Exception as e:
        if session:
            session.rollback()
        raise e
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
    try:
        session = SessionFactory()
        file = session.query(UploadedFile).filter(
            UploadedFile.file_type == file_type
        ).order_by(
            UploadedFile.created_at.desc()
        ).first()

        if file:
            # Convert bytes data to pandas DataFrame
            file_bytes = io.BytesIO(file.file_data)
            df = pd.read_excel(file_bytes)
            return (file.filename, df)
        else:
            return None
    except Exception as e:
        raise e
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
        # Create a database engine
        engine = create_engine(DATABASE_URL) #Use existing engine

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
        new_forecast = {
            'id': forecast_id,
            'sku': sku,
            'model_type': model_type,
            'forecast_date': datetime.now(),
            'forecast_periods': forecast_periods,
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'forecast_data': forecast_data,
            'model_params': model_params
        }

        # Insert the record using text() to prevent SQL injection vulnerabilities.
        with engine.connect() as conn:
            conn.execute(
                text("""
                INSERT INTO forecast_results 
                (id, sku, model_type, forecast_date, forecast_periods, mape, rmse, mae, forecast_data, model_params)
                VALUES 
                (:id, :sku, :model_type, :forecast_date, :forecast_periods, :mape, :rmse, :mae, :forecast_data, :model_params)
                """),
                new_forecast
            )
            conn.commit()

        return True

    except Exception as e:
        print(f"Error saving forecast to database: {str(e)}")
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
            return {
                'parameters': json.loads(cache.parameters),
                'last_updated': cache.last_updated,
                'tuning_iterations': cache.tuning_iterations,
                'best_score': cache.best_score
            }
        
        return None
    
    except Exception as e:
        print(f"Error retrieving model parameters: {str(e)}")
        return None
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
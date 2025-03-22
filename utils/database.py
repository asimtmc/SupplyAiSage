import os
import io
import uuid
import base64
import pandas as pd
from sqlalchemy import create_engine, Column, String, LargeBinary, DateTime, Text, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

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
        if file_type_exists(file_type):
            # Find the existing file(s) of this type and delete them
            existing_files = session.query(UploadedFile).filter(
                UploadedFile.file_type == file_type
            ).all()
            
            for existing_file in existing_files:
                session.delete(existing_file)
            
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
    Save a forecast result to the database
    
    Parameters:
    -----------
    sku : str
        SKU identifier
    model_type : str
        Type of forecasting model used
    forecast_periods : int
        Number of periods forecasted
    mape : float
        Mean Absolute Percentage Error
    rmse : float
        Root Mean Square Error
    mae : float
        Mean Absolute Error
    forecast_data : str
        JSON string of forecast values
    model_params : str
        JSON string of model parameters
    
    Returns:
    --------
    str
        ID of the saved forecast result
    """
    try:
        session = SessionFactory()
        
        # Generate a unique ID
        forecast_id = str(uuid.uuid4())
        
        # Create a new ForecastResult record
        new_forecast = ForecastResult(
            id=forecast_id,
            sku=sku,
            model_type=model_type,
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
        
        return forecast_id
    except Exception as e:
        if session:
            session.rollback()
        raise e
    finally:
        if session:
            session.close()

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
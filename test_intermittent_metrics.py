import pandas as pd
import numpy as np
from utils.forecast_engine import calculate_intermittent_metrics
import streamlit as st

def main():
    st.title("Test Intermittent Demand Metrics")
    
    # Create sample data with intermittent pattern
    st.header("Sample Data with Intermittent Pattern")
    
    # Regular demand pattern
    actual_regular = np.array([10, 12, 15, 11, 14, 13, 16, 12, 15, 17, 14, 16])
    forecast_regular = np.array([11, 13, 14, 12, 13, 14, 15, 13, 14, 16, 15, 15])
    
    # Intermittent demand pattern
    actual_intermittent = np.array([0, 15, 0, 0, 20, 0, 0, 0, 18, 0, 0, 25])
    forecast_intermittent = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    
    # Display sample data
    df_regular = pd.DataFrame({
        'Actual': actual_regular,
        'Forecast': forecast_regular
    })
    
    df_intermittent = pd.DataFrame({
        'Actual': actual_intermittent,
        'Forecast': forecast_intermittent
    })
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Regular Demand Pattern")
        st.dataframe(df_regular)
    
    with col2:
        st.subheader("Intermittent Demand Pattern")
        st.dataframe(df_intermittent)
    
    # Calculate metrics using both standard and intermittent approaches
    st.header("Metrics Comparison")
    
    # Standard metrics for regular data
    standard_rmse_regular = np.sqrt(np.mean((actual_regular - forecast_regular) ** 2))
    mask_regular = actual_regular > 0
    standard_mape_regular = np.mean(np.abs((actual_regular[mask_regular] - forecast_regular[mask_regular]) / actual_regular[mask_regular])) * 100
    standard_mae_regular = np.mean(np.abs(actual_regular - forecast_regular))
    
    # Standard metrics for intermittent data (will have issues)
    standard_rmse_intermittent = np.sqrt(np.mean((actual_intermittent - forecast_intermittent) ** 2))
    mask_intermittent = actual_intermittent > 0
    try:
        standard_mape_intermittent = np.mean(np.abs((actual_intermittent[mask_intermittent] - forecast_intermittent[mask_intermittent]) / actual_intermittent[mask_intermittent])) * 100
    except:
        standard_mape_intermittent = "Error - division by zero"
    standard_mae_intermittent = np.mean(np.abs(actual_intermittent - forecast_intermittent))
    
    # Specialized intermittent metrics for both
    intermittent_metrics_regular = calculate_intermittent_metrics(actual_regular, forecast_regular)
    intermittent_metrics_intermittent = calculate_intermittent_metrics(actual_intermittent, forecast_intermittent)
    
    # Display results in a table
    metrics_data = {
        'Metric': ['RMSE', 'MAPE (%)', 'MAE'],
        'Regular Data (Standard)': [
            f"{standard_rmse_regular:.2f}", 
            f"{standard_mape_regular:.2f}%", 
            f"{standard_mae_regular:.2f}"
        ],
        'Regular Data (Intermittent Metrics)': [
            f"{intermittent_metrics_regular['rmse']:.2f}", 
            f"{intermittent_metrics_regular['mape']:.2f}%" if not np.isnan(intermittent_metrics_regular['mape']) else "N/A", 
            f"{intermittent_metrics_regular['mae']:.2f}"
        ],
        'Intermittent Data (Standard)': [
            f"{standard_rmse_intermittent:.2f}", 
            standard_mape_intermittent if isinstance(standard_mape_intermittent, str) else f"{standard_mape_intermittent:.2f}%", 
            f"{standard_mae_intermittent:.2f}"
        ],
        'Intermittent Data (Intermittent Metrics)': [
            f"{intermittent_metrics_intermittent['rmse']:.2f}", 
            f"{intermittent_metrics_intermittent['mape']:.2f}%" if not np.isnan(intermittent_metrics_intermittent['mape']) else "N/A", 
            f"{intermittent_metrics_intermittent['mae']:.2f}"
        ]
    }
    
    st.dataframe(pd.DataFrame(metrics_data))
    
    st.header("Explanation")
    st.markdown("""
        ### What's happening here?
        
        For regular demand patterns, both standard and specialized metrics give similar results.
        
        However, with intermittent patterns:
        
        1. **Standard MAPE** struggles with zeros in the data and produces errors or extreme values.
        2. **Specialized Metrics** use cumulative values to provide more stable and interpretable metrics:
           - Calculates metrics based on cumulative totals
           - Handles zero values gracefully
           - Provides values that can be fairly compared across all models
        
        This is why specialized metrics are crucial for intermittent demand forecasting models like Croston, SBA, and TSB.
    """)

if __name__ == "__main__":
    main()
# Intermittent Demand Forecasting Platform

A specialized forecasting application optimized for products with intermittent demand patterns, using the Croston method and other advanced forecasting techniques.

## Features

- **Specialized for Intermittent Demand**: Automatically detects and handles demand patterns with frequent zero values
- **Croston Method Implementation**: Uses the industry-standard approach for intermittent demand forecasting
- **Automatic Seasonal Period Detection**: Intelligently identifies seasonal patterns in your data
- **Hyperparameter Tuning**: Optimizes model parameters for better accuracy
- **Interactive Visualizations**: Explore forecasts with dynamic charts

## Getting Started

### Local Development

1. Install requirements:
```
pip install -r requirements-minimal.txt
```

2. Initialize the database:
```
python init_db.py
```

3. Run the application:
```
streamlit run app.py
```

### Heroku Deployment

This application is optimized for deployment on Heroku with a size limit of 450MB.

1. Create a new Heroku app:
```
heroku create your-app-name
```

2. Set Python runtime:
```
heroku buildpacks:set heroku/python
```

3. Push to Heroku:
```
git push heroku main
```

4. Run database initialization:
```
heroku run python init_db.py
```

## Data Format

The application expects sales data with at least the following columns:
- `date`: Date of sale (YYYY-MM-DD format)
- `sku`: Product identifier
- `quantity`: Quantity sold

## Optimization for Heroku

The application has been significantly trimmed to focus exclusively on the core forecasting functionality to meet Heroku's 450MB slug size limit:

- Removed TensorFlow and other large dependencies
- Simplified to focus exclusively on Croston method
- Optimized database operations
- Reduced code duplication

## File Structure

- `/utils`: Core utilities for forecasting and data processing
- `/pages`: Streamlit pages for the web interface
- `app.py`: Main application entry point
- `init_db.py`: Database initialization script
- `heroku_startup.py`: Automated setup for Heroku deployment
- `.slugignore`: Configuration to exclude unnecessary files during deployment

## License

© 2025 Intermittent Demand Forecasting Platform
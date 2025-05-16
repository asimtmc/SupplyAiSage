# AI Supply Chain Platform

Advanced AI-powered supply chain forecasting and analysis tool with time series analysis, seasonal decomposition, and multiple forecasting models.

## Features

- Demand Forecasting with multiple algorithms (SARIMA, Prophet, Neural Networks)
- Production Planning and Optimization
- Materials Procurement Scheduling
- Interactive Dashboards & What-If Scenarios
- Specialized handling for intermittent demand patterns
- Detailed transition management and material depletion tracking
- Database Viewer with detailed row exploration and export functionality

## Deployment to Heroku

### Prerequisites

- Heroku account
- Heroku CLI installed
- Git installed

### Steps to Deploy

1. **Login to Heroku**
   ```
   heroku login
   ```

2. **Create a Heroku app**
   ```
   heroku create your-app-name
   ```

3. **Add PostgreSQL addon**
   ```
   heroku addons:create heroku-postgresql:mini
   ```

4. **Push to Heroku**
   ```
   git push heroku main
   ```

5. **Ensure at least one dyno is running**
   ```
   heroku ps:scale web=1
   ```

6. **Open the app in browser**
   ```
   heroku open
   ```

### Environment Variables

The application will automatically use the `DATABASE_URL` environment variable provided by Heroku.

### Troubleshooting

If you encounter issues with deployment:

1. Check Heroku logs:
   ```
   heroku logs --tail
   ```

2. Ensure Python 3.12 is properly specified in `.python-version`

3. Verify your Procfile is correctly configured

## Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements-deploy.txt`
3. Run the application: `streamlit run app.py`

## Database

The application uses SQLite in local development and PostgreSQL in production.
All database tables are automatically created when the application starts.

## File Structure

- `app.py`: Main application entry point
- `pages/`: Streamlit pages for different modules
- `utils/`: Utility functions and database operations
- `data/`: Directory for storing the SQLite database and cache files
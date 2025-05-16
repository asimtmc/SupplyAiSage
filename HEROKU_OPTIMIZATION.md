# Heroku Deployment Optimization Guide

This document provides a guide for optimizing the Intermittent Demand Forecasting application for deployment on Heroku, which has a 450MB slug size limit.

## Current Optimizations

1. **Focused Functionality**: Application is trimmed to focus exclusively on the V2 Demand Forecasting with Croston method.
2. **Reduced Dependencies**: Only essential packages are included in requirements.
3. **Optimized Database**: SQLite with minimal table structure.
4. **Removed Large Libraries**: TensorFlow and other large dependencies have been removed.
5. **Automated Cleanup**: Scripts to remove unnecessary files before deployment.

## Size Reduction Steps

### 1. Clean Up Unused Pages and Modules

Run the provided cleanup script to safely remove unused pages while backing up everything:

```bash
./bin/cleanup_for_heroku.sh
```

This script:
- Creates a backup of all files before removal
- Keeps only the essential pages and modules
- Removes cache files and directories

### 2. Use the Correct Requirements File

For Heroku deployment, use the optimized requirements file:

```bash
cp heroku_prod_requirements.txt requirements.txt
```

### 3. Optimize Database

Before deployment, optimize the database:

```bash
sqlite3 data/supply_chain.db "VACUUM;"
```

### 4. Heroku-specific Configurations

The following files have been configured for Heroku:

- `Procfile`: Specifies the command to run the application
- `runtime.txt`: Specifies Python 3.10.12
- `.slugignore`: Tells Heroku which files to ignore during deployment
- `app.json`: Provides metadata for the Heroku app

### 5. Final Size Check

Before deploying, you can check the estimated slug size by running:

```bash
tar -cz --exclude=".git" --exclude="venv" . | wc -c
```

This should report a size less than 450MB (471,859,200 bytes).

## Troubleshooting Heroku Deployments

1. **H14 - No Web Dynos Running**: Make sure your Procfile is in the root directory and contains `web: streamlit run app.py --server.port=$PORT --server.headless=true`

2. **H10 - App Crashed**: Check the logs using `heroku logs --tail` to identify the issue.

3. **R14 - Memory Quota Exceeded**: The app is using too much memory. Consider further optimizations or upgrading your Heroku plan.

4. **Slug Size Too Large**:
   - Run `heroku apps:info` to check the slug size
   - Use the cleanup script to remove more files
   - Check for large data files in the repository

## Maintaining Size Optimization

When adding new features:

1. Avoid large dependencies
2. Keep data processing efficient
3. Use the `.slugignore` file to exclude development-only files
4. Run the optimization scripts before each deployment

By following these guidelines, you should be able to keep the application under the 450MB limit while maintaining full functionality.
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def analyze_forecast_errors(actual, predicted, dates=None):
    """
    Analyze forecast errors in detail to identify patterns and issues
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    dates : array-like, optional
        Dates or time points corresponding to values
        
    Returns:
    --------
    dict
        Dictionary with error analysis results
    """
    # Convert inputs to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate basic errors
    errors = actual - predicted
    abs_errors = np.abs(errors)
    
    # Calculate percentage errors (handling zeros)
    if (actual == 0).any():
        epsilon = 1e-10
        percent_errors = 100 * abs_errors / (actual + epsilon)
    else:
        percent_errors = 100 * abs_errors / actual
    
    # Create DataFrame for analysis if dates are provided
    if dates is not None:
        error_df = pd.DataFrame({
            'date': dates,
            'actual': actual,
            'predicted': predicted,
            'error': errors,
            'abs_error': abs_errors,
            'percent_error': percent_errors
        })
    else:
        error_df = pd.DataFrame({
            'actual': actual,
            'predicted': predicted,
            'error': errors,
            'abs_error': abs_errors,
            'percent_error': percent_errors
        })
    
    # Basic error statistics
    error_stats = {
        'mean_error': np.mean(errors),
        'mean_abs_error': np.mean(abs_errors),
        'mean_pct_error': np.mean(percent_errors),
        'median_abs_error': np.median(abs_errors),
        'std_error': np.std(errors),
        'max_error': np.max(abs_errors),
        'min_error': np.min(abs_errors),
        'rmse': np.sqrt(np.mean(np.square(errors))),
        'mape': np.mean(percent_errors)
    }
    
    # Error distributions
    q25, q50, q75 = np.percentile(abs_errors, [25, 50, 75])
    error_dist = {
        'q25': q25,
        'median': q50,
        'q75': q75,
        'iqr': q75 - q25,
        'histogram': np.histogram(abs_errors, bins=10)
    }
    
    # Check for bias in errors
    bias_analysis = {
        'bias': np.mean(errors),
        'bias_pct': 100 * np.mean(errors) / np.mean(actual) if np.mean(actual) != 0 else np.nan,
        'positive_errors': np.sum(errors > 0),
        'negative_errors': np.sum(errors < 0),
        'error_symmetry': np.sum(errors > 0) / len(errors) if len(errors) > 0 else 0.5
    }
    
    # Calculate autocorrelation of errors if enough data points
    error_autocorr = None
    if len(errors) >= 4:
        try:
            from statsmodels.tsa.stattools import acf
            error_autocorr = acf(errors, nlags=min(10, len(errors) // 2))
        except:
            # If statsmodels not available, calculate a simple lag-1 autocorrelation
            error_autocorr = np.corrcoef(errors[:-1], errors[1:])[0, 1] if len(errors) > 1 else 0
    
    # Evaluate error patterns
    patterns = {
        'systematic_bias': abs(bias_analysis['bias_pct']) > 10,
        'autocorrelation': error_autocorr[1] > 0.3 if isinstance(error_autocorr, np.ndarray) and len(error_autocorr) > 1 else False,
        'extreme_errors': np.sum(abs_errors > 2 * error_stats['mean_abs_error']) / len(abs_errors) if len(abs_errors) > 0 else 0
    }
    
    # Create overall error analysis report
    analysis = {
        'error_stats': error_stats,
        'error_distribution': error_dist,
        'bias_analysis': bias_analysis,
        'autocorrelation': error_autocorr,
        'patterns': patterns,
        'error_details': error_df
    }
    
    return analysis

def analyze_model_performance(models_dict, actual_data):
    """
    Analyze and compare performance of multiple models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and forecast results as values
    actual_data : array-like
        Actual values to compare against forecasts
        
    Returns:
    --------
    dict
        Dictionary with model performance analysis
    """
    if not models_dict:
        return None
    
    # Prepare actual data
    actual = np.array(actual_data)
    
    # Calculate metrics for each model
    model_metrics = {}
    model_errors = {}
    
    for model_name, forecast_data in models_dict.items():
        if forecast_data is None or 'test' not in forecast_data:
            continue
            
        # Get forecast values
        predicted = np.array(forecast_data['test'])
        
        # Check if lengths match
        if len(predicted) != len(actual):
            continue
            
        # Calculate basic metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # Calculate MAPE
        if (actual == 0).any():
            epsilon = 1e-10
            mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
        else:
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Store metrics
        model_metrics[model_name] = {'mae': mae, 'rmse': rmse, 'mape': mape}
        
        # Store detailed error analysis
        model_errors[model_name] = analyze_forecast_errors(actual, predicted)
    
    # Rank models based on different metrics
    metric_rankings = {}
    
    for metric in ['mae', 'rmse', 'mape']:
        # Get valid models for this metric
        valid_models = {m: metrics[metric] for m, metrics in model_metrics.items() 
                      if metric in metrics and not np.isnan(metrics[metric])}
        
        # Rank models by this metric (lower is better)
        ranked_models = sorted(valid_models.items(), key=lambda x: x[1])
        
        metric_rankings[metric] = {model: rank+1 for rank, (model, _) in enumerate(ranked_models)}
    
    # Create overall rankings by averaging rank across metrics
    overall_ranks = {}
    
    for model in model_metrics:
        # Get ranks for this model across all metrics
        model_ranks = [rankings.get(model, len(model_metrics)) 
                     for metric, rankings in metric_rankings.items()]
        
        # Calculate average rank
        avg_rank = np.mean(model_ranks)
        
        overall_ranks[model] = avg_rank
    
    # Create performance summary
    performance_summary = {
        'metrics': model_metrics,
        'detailed_errors': model_errors,
        'rankings': metric_rankings,
        'overall_ranking': overall_ranks,
        'best_model': min(overall_ranks.items(), key=lambda x: x[1])[0] if overall_ranks else None
    }
    
    return performance_summary

def generate_error_report(analysis_result, model_name=None):
    """
    Generate a human-readable report from error analysis
    
    Parameters:
    -----------
    analysis_result : dict
        Dictionary with error analysis results from analyze_forecast_errors
    model_name : str, optional
        Name of the model to include in the report
        
    Returns:
    --------
    str
        Text report with error analysis findings
    """
    if not analysis_result:
        return "No error analysis data available."
    
    # Basic stats
    error_stats = analysis_result.get('error_stats', {})
    patterns = analysis_result.get('patterns', {})
    bias = analysis_result.get('bias_analysis', {})
    
    # Create report
    report = []
    
    # Model header
    if model_name:
        report.append(f"ERROR ANALYSIS REPORT FOR: {model_name.upper()}")
    else:
        report.append("FORECAST ERROR ANALYSIS REPORT")
    
    report.append("-" * 50)
    
    # Summary metrics
    report.append("SUMMARY METRICS:")
    report.append(f"  MAPE: {error_stats.get('mape', 'N/A'):.2f}%")
    report.append(f"  RMSE: {error_stats.get('rmse', 'N/A'):.2f}")
    report.append(f"  MAE: {error_stats.get('mean_abs_error', 'N/A'):.2f}")
    report.append("")
    
    # Error patterns
    report.append("ERROR PATTERNS:")
    
    # Bias
    bias_pct = bias.get('bias_pct', 0)
    if abs(bias_pct) > 10:
        report.append(f"  ⚠️ Systematic Bias: {bias_pct:.1f}% {'overestimation' if bias_pct < 0 else 'underestimation'}")
    else:
        report.append(f"  ✓ No significant bias detected ({bias_pct:.1f}%)")
    
    # Autocorrelation
    autocorr = analysis_result.get('autocorrelation', None)
    if isinstance(autocorr, np.ndarray) and len(autocorr) > 1 and autocorr[1] > 0.3:
        report.append(f"  ⚠️ Error Autocorrelation: Lag-1 = {autocorr[1]:.2f} (errors are serially correlated)")
    else:
        report.append("  ✓ No significant error autocorrelation")
    
    # Extreme errors
    extreme_error_rate = patterns.get('extreme_errors', 0)
    if extreme_error_rate > 0.1:
        report.append(f"  ⚠️ Extreme Errors: {extreme_error_rate*100:.1f}% of predictions have large errors (>2x MAE)")
    else:
        report.append(f"  ✓ Errors are consistent without many extreme values ({extreme_error_rate*100:.1f}%)")
    
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS:")
    if abs(bias_pct) > 10:
        report.append("  • Investigate systematic bias in the forecast - check for missing factors or data preprocessing issues")
    
    if isinstance(autocorr, np.ndarray) and len(autocorr) > 1 and autocorr[1] > 0.3:
        report.append("  • Model is missing temporal patterns - consider additional features or different model type")
    
    if extreme_error_rate > 0.1:
        report.append("  • Improve outlier handling or explore ensemble methods to reduce extreme errors")
    
    # Generate overall assessment
    issues_count = sum(1 for v in patterns.values() if v)
    if issues_count == 0:
        overall = "✓ GOOD: No significant issues detected in forecast errors"
    elif issues_count == 1:
        overall = "⚠️ FAIR: Minor issues detected in forecast errors"
    else:
        overall = "⚠️ NEEDS IMPROVEMENT: Multiple issues detected in forecast errors"
    
    report.append("")
    report.append("OVERALL ASSESSMENT:")
    report.append(f"  {overall}")
    
    return "\n".join(report)
"""
Configuration for active pages in the optimized deployment.
Only pages listed here will be available in the application.
"""

# List of active page files (only these will be included in the build)
ACTIVE_PAGES = [
    "14_V2_Demand_Forecasting_Croston.py"
]

# Mapping of page IDs to their display names
PAGE_NAMES = {
    "14_V2_Demand_Forecasting_Croston.py": "Intermittent Demand Forecasting"
}

def is_page_active(page_filename):
    """Check if a page should be active in the current deployment."""
    return page_filename in ACTIVE_PAGES

def get_page_name(page_filename):
    """Get the display name for a page."""
    return PAGE_NAMES.get(page_filename, "Unknown Page")
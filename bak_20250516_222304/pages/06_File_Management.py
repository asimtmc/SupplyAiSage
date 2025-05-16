
import streamlit as st
import pandas as pd
import io
import json
from datetime import datetime
from streamlit_extras.colored_header import colored_header
from streamlit_extras.grid import grid
from streamlit_card import card
from utils.database import get_all_files, get_file_by_id, delete_file

st.set_page_config(page_title="File Management", page_icon="📂", layout="wide")

# Add custom CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .file-card {
        border: 1px solid #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    .file-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    .highlight {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #4CAF50;
        margin: 1rem 0;
    }
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #45a049;
    }
    .delete-btn {
        background-color: #f44336 !important;
    }
    .delete-btn:hover {
        background-color: #d32f2f !important;
    }
</style>
""", unsafe_allow_html=True)

colored_header(
    label="File Management",
    description="Manage your uploaded data files",
    color_name="green-70",
)

# Initialize session state for file management
if 'selected_file_id' not in st.session_state:
    st.session_state.selected_file_id = None

# Create tabs
tab1, tab2 = st.tabs(["📋 File Library", "📊 File Details"])

with tab1:
    st.subheader("Uploaded Files")
    
    # Get all files from database
    try:
        files = get_all_files()
        
        if not files:
            st.info("No files have been uploaded yet. Upload files from the respective modules.")
        else:
            # Group files by type
            file_types = {}
            for file in files:
                file_type = file['file_type']
                if file_type not in file_types:
                    file_types[file_type] = []
                file_types[file_type].append(file)
            
            # Create file type sections with cards
            for file_type, type_files in file_types.items():
                st.subheader(f"{file_type.replace('_', ' ').title()}")
                
                # Create a grid for cards
                file_grid = grid(3, vertical_align="top")
                
                # Add cards for each file
                for i, file in enumerate(type_files):
                    with file_grid.container():
                        # Format the date
                        date_obj = datetime.strptime(file['created_at'], '%Y-%m-%d %H:%M:%S')
                        formatted_date = date_obj.strftime('%b %d, %Y')
                        
                        # Create a card for the file with a unique key
                        # Fix: Add unique key to avoid duplicate component instance error
                        card_key = f"{file_type}_{i}_{file['id']}"
                        clicked = card(
                            title=file['filename'],
                            text=[
                                f"Type: {file['file_type'].replace('_', ' ').title()}",
                                f"Uploaded: {formatted_date}",
                                "📥 Click to view/download"
                            ],
                            image=None,
                            url=None,
                            key=card_key,  # Add unique key here
                            on_click=lambda f_id=file['id']: st.session_state.update({'selected_file_id': f_id})
                        )
                        
                        if clicked:
                            st.experimental_rerun()
                
                st.markdown("---")
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")

with tab2:
    if st.session_state.selected_file_id:
        try:
            # Get file details
            file_result = get_file_by_id(st.session_state.selected_file_id)
            
            if file_result:
                filename, file_data = file_result
                
                # Display file details
                st.subheader(f"File: {filename}")
                
                # Determine file type and MIME type
                mime_type = "application/octet-stream"  # Default MIME type
                if filename.endswith(('.xlsx', '.xls')):
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif filename.endswith('.csv'):
                    mime_type = "text/csv"
                elif filename.endswith('.json'):
                    mime_type = "application/json"
                elif filename.endswith('.txt'):
                    mime_type = "text/plain"
                elif filename.endswith('.pdf'):
                    mime_type = "application/pdf"
                
                # Display file preview for supported file types
                preview_col, download_col = st.columns([3, 1])
                
                with preview_col:
                    if filename.endswith(('.xlsx', '.xls')):
                        try:
                            df = pd.read_excel(io.BytesIO(file_data))
                            st.dataframe(df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error previewing Excel file: {str(e)}")
                    elif filename.endswith('.csv'):
                        try:
                            df = pd.read_csv(io.BytesIO(file_data))
                            st.dataframe(df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error previewing CSV file: {str(e)}")
                    elif filename.endswith('.json'):
                        try:
                            data_str = file_data.decode('utf-8')
                            json_data = json.loads(data_str)
                            st.json(json_data)
                        except Exception as e:
                            st.error(f"Error previewing JSON file: {str(e)}")
                    elif filename.endswith('.txt'):
                        try:
                            st.text(file_data.decode('utf-8'))
                        except Exception as e:
                            st.error(f"Error previewing text file: {str(e)}")
                    else:
                        st.info("File preview not available for this file type")
                
                with download_col:
                    # Download button with appropriate styling
                    st.markdown("<h3>Download Options</h3>", unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download File",
                        data=file_data,
                        file_name=filename,
                        mime=mime_type,
                        key="download_btn",
                        help="Download this file to your computer",
                        use_container_width=True
                    )
                
                # Delete button
                if st.button("🗑️ Delete File", key="delete_btn", help="Delete this file permanently", type="primary", use_container_width=True):
                    if delete_file(st.session_state.selected_file_id):
                        st.session_state.selected_file_id = None
                        st.success("File deleted successfully")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to delete file")
            else:
                st.warning("File not found. It may have been deleted.")
        except Exception as e:
            st.error(f"Error loading file details: {str(e)}")
    else:
        st.info("Select a file from the File Library tab to view details")

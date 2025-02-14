import streamlit as st
import requests
import pandas as pd
from io import StringIO
from datetime import datetime
import urllib3
import socket

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration and Setup
st.set_page_config(layout="wide")

# Constants and Configuration
class Config:
    # URL Configuration
    BASE_URL = "http://172.16.0.207/OTRS_system/Download11.php?process=xidops"
    
    # Request Configuration
    REQUEST_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    
    # VPN Network Configuration
    VERIFY_SSL = False  # Set to False when using self-signed certificates
    PROXY_SETTINGS = None  # Set to None to use system proxy settings
    
    @staticmethod
    def get_session():
        session = requests.Session()
        # Configure session for VPN
        session.verify = Config.VERIFY_SSL
        session.proxies = Config.PROXY_SETTINGS
        
        # Configure retry strategy
        adapter = requests.adapters.HTTPAdapter(
            max_retries=Config.MAX_RETRIES,
            pool_connections=10,
            pool_maxsize=10
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session

REQUIRED_COLUMNS = ["username", "resolved", "first_closed_ticket_timestamp"]

# Custom CSS
st.markdown(
    """
    <style>
    .header-section {
        background-color: #f8f9fa;
        padding: 1rem 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .custom-title {
        font-size: 32px;
        font-weight: bold;
        color: #4CAF50;
        margin: 0;
        padding: 0;
    }
    .table {
        font-size: 12px;
    }
    .stDateInput {
        width: 180px !important;
    }
    div[data-baseweb="input"] {
        width: 180px !important;
    }
    .stDateInput > label {
        font-size: 14px !important;
        color: #666;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a container for the header section
st.markdown('<div class="header-section">', unsafe_allow_html=True)

# Layout for title, date selectors, and fetch button
header_cols = st.columns([2, 1.2, 1.2, 1])

with header_cols[0]:
    st.markdown('<h1 class="custom-title">T3 Count Viewer</h1>', unsafe_allow_html=True)

with header_cols[1]:
    start_date = st.date_input(
        "Start date",
        min_value=datetime(2000, 1, 1),
        max_value=datetime.today(),
        value=datetime.today(),
        key="start_date"
    )

with header_cols[2]:
    end_date = st.date_input(
        "End date",
        min_value=start_date,
        max_value=datetime.today(),
        value=datetime.today(),
        key="end_date"
    )

with header_cols[3]:
    st.write("")  # Add some vertical spacing
    fetch_data_button = st.button("Fetch Data")

st.markdown('</div>', unsafe_allow_html=True)

# Convert start_date and end_date to datetime with time set to start and end of day
start_datetime = pd.Timestamp(start_date).replace(hour=0, minute=0, second=0)
end_datetime = pd.Timestamp(end_date).replace(hour=23, minute=59, second=59)

selected_dates = pd.date_range(start=start_date, end=end_date).strftime('%d-%b-%Y').tolist()
date_heading = f"Resolved Tickets Breakdown ({start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')})"

def fetch_and_process_data(url):
    """Fetch and process data from the URL with VPN support."""
    try:
        session = Config.get_session()
        response = session.get(
            url,
            timeout=Config.REQUEST_TIMEOUT,
            verify=Config.VERIFY_SSL
        )
        response.raise_for_status()
        
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, delimiter=',', quotechar='"', skip_blank_lines=True, on_bad_lines='skip')
        
        if df.shape[1] == 24:
            df = df.iloc[:, :23]
        
        if all(col in df.columns for col in REQUIRED_COLUMNS):
            df_cleaned = df[REQUIRED_COLUMNS]
            df_cleaned['resolved'] = pd.to_numeric(df_cleaned['resolved'], errors='coerce')
            df_cleaned['first_closed_ticket_timestamp'] = pd.to_datetime(df_cleaned['first_closed_ticket_timestamp'], errors='coerce')
            return df_cleaned
        else:
            raise ValueError("Required columns missing in dataset")
            
    except requests.exceptions.SSLError:
        st.error("SSL Certificate verification failed. Please check your VPN connection and certificate settings.")
        raise
    except requests.exceptions.ProxyError:
        st.error("Unable to connect through proxy. Please check your VPN connection.")
        raise
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the server. Please verify your VPN connection is active and working.")
        raise
    except socket.timeout:
        st.error(f"Request timed out after {Config.REQUEST_TIMEOUT} seconds. Please check your network connection.")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        raise

def generate_hourly_summary(df, start_datetime, end_datetime):
    """Generate hourly breakdown of resolved tickets."""
    df = df[df['resolved'] == 1].copy()
    df['first_closed_ticket_timestamp'] = pd.to_datetime(df['first_closed_ticket_timestamp'], errors='coerce')
    
    # Filter by date range
    mask = (df['first_closed_ticket_timestamp'] >= start_datetime) & (df['first_closed_ticket_timestamp'] <= end_datetime)
    df = df[mask]
    
    df['Hour'] = df['first_closed_ticket_timestamp'].dt.strftime('%H:00')
    
    hourly_summary = df.pivot_table(
        index=['username'], 
        columns=['Hour'], 
        values='resolved', 
        aggfunc='count', 
        fill_value=0
    )
    
    hourly_summary['Total'] = hourly_summary.sum(axis=1)
    total_row = pd.DataFrame(hourly_summary.sum(axis=0)).T
    total_row.index = ['Total']
    hourly_summary = pd.concat([hourly_summary, total_row])
    
    cols = ['Total'] + [col for col in hourly_summary.columns if col != 'Total']
    return hourly_summary[cols]

def process_open_tickets(df):
    """Process open tickets data."""
    open_tickets = df[df['resolved'].isin([0, 2])]
    open_tickets_summary = open_tickets.groupby('username').size().reset_index(name='Open tickets')
    total_open_tickets = open_tickets_summary['Open tickets'].sum()
    open_tickets_summary.loc[len(open_tickets_summary)] = ['Total', total_open_tickets]
    return open_tickets_summary, total_open_tickets

def process_resolved_tickets(df, start_datetime, end_datetime):
    """Process resolved tickets data with correct filtering and counting."""
    # First filter for resolved=1 tickets
    resolved_df = df[df['resolved'] == 1].copy()
    
    # Convert timestamp and filter by date range
    resolved_df['first_closed_ticket_timestamp'] = pd.to_datetime(resolved_df['first_closed_ticket_timestamp'])
    mask = ((resolved_df['first_closed_ticket_timestamp'] >= start_datetime) & 
            (resolved_df['first_closed_ticket_timestamp'] <= end_datetime))
    resolved_df = resolved_df[mask]
    
    # Format date for display
    resolved_df['date'] = resolved_df['first_closed_ticket_timestamp'].dt.strftime('%d-%b-%Y')
    
    # Create pivot table with correct counting
    resolved_summary = pd.pivot_table(
        resolved_df,
        index='username',
        columns='date',
        values='resolved',
        aggfunc='count',
        fill_value=0
    ).reset_index()
    
    # Calculate total and reorder columns
    resolved_summary['Total'] = resolved_summary.iloc[:, 1:].sum(axis=1)
    total_col = resolved_summary.pop('Total')
    resolved_summary.insert(1, 'Total', total_col)
    
    # Add total row
    total_row = pd.Series(
        ['Total'] + resolved_summary.iloc[:, 1:].sum().tolist(),
        index=resolved_summary.columns
    )
    resolved_summary = pd.concat([resolved_summary, total_row.to_frame().T], ignore_index=True)
    
    return resolved_summary

# Main Application Logic
if fetch_data_button:
    try:
        with st.spinner("Fetching and processing data..."):
            # Fetch and process data
            df_cleaned = fetch_and_process_data(Config.BASE_URL)
            
            # Process open tickets
            open_tickets_summary, total_open_tickets = process_open_tickets(df_cleaned)
            
            # Process resolved tickets
            resolved_summary = process_resolved_tickets(df_cleaned.copy(), start_datetime, end_datetime)
            
            # Generate hourly breakdown
            hourly_summary = generate_hourly_summary(df_cleaned.copy(), start_datetime, end_datetime)
            
            # Display results
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"Open Tickets Summary: (Total: {total_open_tickets})")
                st.table(open_tickets_summary)
            
            with col2:
                st.write("Resolved Tickets Count:")
                st.table(resolved_summary)
            
            st.write(date_heading)
            st.table(hourly_summary)
            
            # Export options
            col3, col4 = st.columns(2)
            with col3:
                csv_resolved = resolved_summary.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Resolved Tickets Summary",
                    data=csv_resolved,
                    file_name="resolved_tickets_summary.csv",
                    mime="text/csv"
                )
            
            with col4:
                csv_hourly = hourly_summary.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Hourly Breakdown",
                    data=csv_hourly,
                    file_name="hourly_breakdown.csv",
                    mime="text/csv"
                )
                
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data: {e}")
    except ValueError as e:
        st.error(f"Data processing error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
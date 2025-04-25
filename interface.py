# Build interface
import streamlit as st

# Data
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

# Visualisation
import matplotlib.pyplot as plt

# Load the saved model
from joblib import load

df = pd.read_csv('data/train_cleaned.csv')

# Convert 'txn_YearMonth' to datetime for proper sorting
df['txn_YearMonth'] = pd.to_datetime(df['txn_YearMonth'], format='%Y-%m')

# Calculate average resale price by month and flat type
avg_prices = df.groupby(['txn_YearMonth', 'flat_type', 'planning_area'])['resale_price'].mean().reset_index()

st.title("🏘️ Resale Price Trends")
st.markdown("Explore average resale prices by flat type and planning area over time.")

# Get default selections
default_flat_type = avg_prices['flat_type'].unique()[0]
default_planning_area = avg_prices[avg_prices['flat_type'] == default_flat_type]['planning_area'].unique()[0]

# ---- Sidebar Filters ----
st.sidebar.header("Filters")

# 1. Flat Type (single-select)
selected_type = st.sidebar.selectbox(
    "Select Flat Type:",
    options=sorted(avg_prices['flat_type'].unique()),
    index=0  # Default to first option
)

# 2. Planning Area (dynamic based on flat_type)
available_areas = sorted(avg_prices[avg_prices['flat_type'] == selected_type]['planning_area'].unique())
selected_area = st.sidebar.selectbox(
    "Select Planning Area:",
    options=available_areas,
    index=0  # Default to first option
)

# 3. Date Range Slider
min_date = avg_prices['txn_YearMonth'].min().to_pydatetime()
max_date = avg_prices['txn_YearMonth'].max().to_pydatetime()
date_range = st.sidebar.slider(
    "Select Date Range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM"
)

# ---- Filter Data ----
filtered_data = avg_prices[
    (avg_prices['flat_type'] == selected_type) &
    (avg_prices['planning_area'] == selected_area) &
    (avg_prices['txn_YearMonth'] >= date_range[0]) & 
    (avg_prices['txn_YearMonth'] <= date_range[1])
]

# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    filtered_data['txn_YearMonth'],
    filtered_data['resale_price'],
    linewidth=2,
    color='#4CAF50'  # Green color
)

# Formatting
ax.set_title(f"Resale Price Trend: {selected_type} in {selected_area}")
ax.set_xlabel("Timeline")
ax.set_ylabel("Average Price ($)")
ax.grid(True, linestyle='--', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# Display time period
st.caption(f"Showing data from {date_range[0].strftime('%b %Y')} to {date_range[1].strftime('%b %Y')}")

# Initialize empty DataFrame with columns required by the model
input_data = pd.DataFrame(index=[0], columns=[
    'floor_area_sqft', 'txn_Month', 'year_completed',
       'mrt_nearest_distance', 'Mall_Nearest_Distance',
       'Hawker_Nearest_Distance', 'mid_storey', 'flat_type_1 ROOM', 'flat_type_2 ROOM',
       'flat_type_3 ROOM', 'flat_type_4 ROOM', 'flat_type_5 ROOM',
       'flat_type_EXECUTIVE', 'flat_type_MULTI-GENERATION',
       'flat_model_2-room', 'flat_model_Adjoined flat', 'flat_model_Apartment', 'flat_model_DBSS',
       'flat_model_Improved', 'flat_model_Improved-Maisonette',
       'flat_model_Maisonette', 'flat_model_Model A',
       'flat_model_Model A-Maisonette', 'flat_model_Model A2',
       'flat_model_Multi Generation', 'flat_model_New Generation',
       'flat_model_Premium Apartment', 'flat_model_Premium Apartment Loft',
       'flat_model_Premium Maisonette', 'flat_model_Simplified',
       'flat_model_Standard', 'flat_model_Terrace', 'flat_model_Type S1',
       'flat_model_Type S2', 'planning_area_Ang Mo Kio', 'planning_area_Bedok', 'planning_area_Bishan',
       'planning_area_Bukit Batok', 'planning_area_Bukit Merah',
       'planning_area_Bukit Panjang', 'planning_area_Bukit Timah',
       'planning_area_Changi', 'planning_area_Choa Chu Kang',
       'planning_area_Clementi', 'planning_area_Downtown Core',
       'planning_area_Geylang', 'planning_area_Hougang',
       'planning_area_Jurong East', 'planning_area_Jurong West',
       'planning_area_Kallang', 'planning_area_Marine Parade',
       'planning_area_Novena', 'planning_area_Outram',
       'planning_area_Pasir Ris', 'planning_area_Punggol',
       'planning_area_Queenstown', 'planning_area_Rochor',
       'planning_area_Sembawang', 'planning_area_Sengkang',
       'planning_area_Serangoon', 'planning_area_Tampines',
       'planning_area_Tanglin', 'planning_area_Toa Payoh',
       'planning_area_Western Water Catchment', 'planning_area_Woodlands',
       'planning_area_Yishun'
])

# Input collection section
st.subheader(
    "What HDB unit are we predicting today?",
    divider="rainbow"  # This adds the rainbow line
)

# Creates 2 columns for positioning input fields
col1, col2 = st.columns(2)

# Column 1 inputs

# Year of Completion
input_data.loc[0, 'year_completed'] = col1.number_input("Year of Completion", min_value=1920, max_value=2025, value=2000)

# Expected month of transaction
month_str = col1.selectbox(
    "Transaction Month",
    ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'),
    index=0,    # Set default value to Jan
    placeholder="Select expected month of transaction:",

)
# Convert short form month to datetime and change to corresponding integer to feed into model.
if month_str:
    input_data.loc[0, 'txn_Month'] = datetime.strptime(month_str, '%b').month

# Planning area of resale flat
planning_area = col1.selectbox(
    'Which planning area?',
    sorted(['Jurong West', 'Woodlands', 'Sengkang', 'Tampines', 'Yishun', 'Bedok',
            'Punggol', 'Hougang', 'Ang Mo Kio', 'Choa Chu Kang', 'Bukit Merah',
            'Bukit Batok', 'Bukit Panjang', 'Toa Payoh', 'Pasir Ris', 'Queenstown',
            'Geylang', 'Sembawang', 'Clementi', 'Jurong East', 'Kallang',
            'Serangoon', 'Bishan', 'Novena', 'Marine Parade', 'Outram', 'Rochor',
            'Bukit Timah', 'Changi', 'Downtown Core', 'Tanglin', 'Western Water Catchment']),
    index=0,
    placeholder="Choose an option"
)
input_data.loc[0, f'planning_area_{planning_area}'] = 1 # Set the respective one-hot encoded column to 1 for the picked option.

# Column 2 inputs

# Floor Area in sqft
input_data.loc[0, 'floor_area_sqft'] = col2.number_input('Floor Area (sqft)', min_value=100.0, max_value=5000.0, value=1000.0)

# Flat Type
flat_type = col1.selectbox(
    "Flat Type",
    ('1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'),
    index=3,    # Sets default value to 4 ROOM, being the most common flat_type. (index = 3 in sorted tuple)
    placeholder="Select expected flat type:",
)
input_data.loc[0, f'flat_type_{flat_type}'] = 1

# Flat Model
flat_model = col1.selectbox(
    "Flat Model",
    sorted(['Model A', 'Improved', 'New Generation', 'Premium Apartment',
            'Simplified', 'Apartment', 'Standard', 'Maisonette', 'Model A2',
            'DBSS', 'Model A-Maisonette', 'Adjoined flat', 'Type S1', 'Type S2',
            'Terrace', 'Multi Generation', 'Premium Apartment', 'Loft',
            'Improved-Maisonette', 'Premium Maisonette', '2-room']),
    index=8,    # Sets default value to Model A (index = 8 in sorted tuple)
    placeholder="Select expected flat model:",
)
input_data.loc[0, f'flat_model_{flat_model}'] =1

# Free text boxes
input_data.loc[0, 'mid_storey'] = col2.number_input("Which Floor is the unit on?", min_value=1, max_value=60, value=8)
input_data.loc[0, 'Mall_Nearest_Distance'] = col2.number_input("Distance to nearest Mall (meters)", min_value=1, value=600)
input_data.loc[0, 'mrt_nearest_distance'] = col2.number_input("Distance to nearest MRT (meters)", min_value=1, value=680)
input_data.loc[0, 'Hawker_Nearest_Distance'] = col2.number_input("Distance to nearest Hawker Centre (meters)", min_value=1, value=780)

# Data type conversion
# Convert numeric columns to numeric data type
numeric_cols = ['floor_area_sqft', 'year_completed', 'txn_Month',
                'mrt_nearest_distance', 'Mall_Nearest_Distance', 
                'Hawker_Nearest_Distance', 'mid_storey']

for col in numeric_cols:
    if col in input_data.columns:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

# Fill NA values
input_data = input_data.fillna(0)

# Load prediction model
model = load("jjbmodel.joblib")

# Prediction calculation happens when 'Generate Now' button is pressed.
# Output predicted resale price and psf
if st.button('Generate now'):
    predicted_price = model.predict(input_data)
    psf = predicted_price/input_data.loc[0, 'floor_area_sqft']
    st.success(f'Predicted Pricing: SGD{predicted_price[0]:,.2f}')
    st.markdown(f'''<div style="background-color: #f0f9ef;  /* Peppermint */">
                ✅ <strong>Predicted PSF: SGD{psf[0]:,.2f}/psf</strong></div>''',
                unsafe_allow_html=True)
# Disclaimers
    st.markdown('''*Disclaimer: This tool is a predictive model only and is provided for informational purposes. JJB makes no guarantees regarding the accuracy, reliability, or completeness of the predictions. Users should exercise their own judgment and not rely solely on this model for decision-making.  
                In this model lies a potential error of approximately S$40,000.00.  
                JJB disclaims all liability for any errors, omissions, or outcomes resulting from the use of this model.*''')
    st.markdown('''*This model is based on historical data and assumptions, which may not account for future uncertainties or unforeseen events.  
                Consult a qualified professional before making any decisions based on this model’s outputs.*''')
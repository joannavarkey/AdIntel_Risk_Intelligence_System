# run: streamlit run app.py

# ================================
# ADONMO RWA ANALYSIS APP
# Built by: Joanna Mariam Varkey
# AdOnMo Summer Internship 2026
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import warnings
from io import BytesIO
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# ================================
# APP CONFIGURATION
# ================================
st.set_page_config(
    page_title     = 'AdOnMo RWA Analysis',
    page_icon      = '',
    layout         = 'wide',
    initial_sidebar_state = 'expanded'
)

# ================================
# UI CUSTOMIZATION — ADONMO THEME
# ================================
st.markdown("""
<style>
    /* ── FONTS ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── MAIN BACKGROUND ── */
    .stApp {
        background-color: #F8FAFC;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B2A4A 0%, #0D1B2E 100%) !important;
        border-right: 3px solid #00B4D8 !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Fix sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio > div {
        display: flex !important;
        flex-direction: column !important;
        gap: 4px !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #CBD5E1 !important;
        background: transparent !important;
        border: none !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(0, 180, 216, 0.15) !important;
        color: #00B4D8 !important;
        padding-left: 20px !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: rgba(0, 180, 216, 0.25) !important;
        color: #00B4D8 !important;
        border-left: 3px solid #00B4D8 !important;
    }

    /* Hide radio circle */
    [data-testid="stSidebar"] .stRadio > div > label > div:first-child {
        display: none !important;
    }

    /* ── SIDEBAR TEXT ── */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }

    /* ── TITLES ── */
    /* ── TITLES ── */
    h1 {
        color: #1B2A4A !important;
        font-weight: 800 !important;
        font-size: 42px !important;
        border-bottom: 3px solid #00B4D8 !important;
        padding-bottom: 12px !important;
        margin-bottom: 20px !important;
        letter-spacing: -0.5px !important;
    }
    
    h2 {
        color: #1B2A4A !important;
        font-weight: 700 !important;
        font-size: 28px !important;
        margin-bottom: 12px !important;
    }
    
    h3 {
        color: #1B2A4A !important;
        font-weight: 600 !important;
        font-size: 20px !important;
        margin-bottom: 8px !important;
    }

    /* ── METRIC CARDS ── */
    [data-testid="stMetric"] {
        background: white !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 24px rgba(0, 180, 216, 0.2) !important;
        border-color: #00B4D8 !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #64748B !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    [data-testid="stMetricValue"] {
        color: #1B2A4A !important;
        font-weight: 700 !important;
        font-size: 24px !important;
    }

    /* ── BUTTONS ── */
    .stButton > button {
        background: linear-gradient(135deg, #00B4D8 0%, #0077B6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0, 180, 216, 0.5) !important;
        background: linear-gradient(135deg, #0077B6 0%, #00B4D8 100%) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── DOWNLOAD BUTTONS ── */
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #00B4D8 0%, #0077B6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 21px !important;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0, 180, 216, 0.5) !important;
        background: linear-gradient(135deg, #0077B6 0%, #00B4D8 100%) !important;
    }

    /* ── FILE UPLOADER ── */

    [data-testid="stFileUploader"] {
        background: white !important;
        border: 2px dashed #00B4D8 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(0, 180, 216, 0.03) !important;
        border-color: #0077B6 !important;
    }
    
    /* Fix upload button dark background */
    [data-testid="stFileUploader"] button {
        background: white !important;
        color: #1B2A4A !important;
        border: 2px solid #00B4D8 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: #00B4D8 !important;
        color: white !important;
    }
    
    [data-testid="stFileUploader"] section {
        background: white !important;
        border: none !important;
    }
    
    [data-testid="stFileUploadDropzone"] {
        background: white !important;
    }

    /* ── DATAFRAMES ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stDataFrame"]:hover {
        box-shadow: 0 6px 20px rgba(0, 180, 216, 0.15) !important;
    }

    /* ── ALERT BOXES ── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border-left: 4px solid #00B4D8 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stAlert"]:hover {
        transform: translateX(4px) !important;
    }

    /* ── SELECTBOX ── */
    [data-testid="stSelectbox"] > div > div {
        background: white !important;
        border-radius: 10px !important;
        border: 1.5px solid #E2E8F0 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stSelectbox"] > div > div:hover {
        border-color: #00B4D8 !important;
        box-shadow: 0 0 0 3px rgba(0, 180, 216, 0.15) !important;
    }

    /* ── SPINNER ── */
    .stSpinner > div {
        border-top-color: #00B4D8 !important;
    }

    /* ── DIVIDER ── */
    hr {
        border: none !important;
        border-top: 1px solid #E2E8F0 !important;
        margin: 24px 0 !important;
    }

    /* ── HIDE STREAMLIT FOOTER ── */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* ── MAIN CONTENT PADDING ── */
    .main .block-container {
        padding: 2rem 3rem !important;
        max-width: 1200px !important;
    }

    /* ── GENERAL TEXT ── */
    p, span, div {
        font-size: 14px !important;
        line-height: 1.6 !important;
        color: #374151 !important;
    }

    /* Fix upload box icon */
    [data-testid="stFileUploader"] svg {
        display: none !important;
    }
    
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }

</style>
""", unsafe_allow_html=True)

# ================================
# HELPER FUNCTION — STYLED HEADING
# ================================
def styled_heading(text, size=18):
    st.markdown(f"""
    <p style="
        color: #00B4D8;
        font-size: {size}px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 16px 0 8px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #E2E8F0;
        ">
        {text}
    </p>
    """, unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================
st.sidebar.image('AdIntel_logo.png', width=300)
st.sidebar.title(' Hi team, Welcome to AdIntel! ')
st.sidebar.markdown('---')
page = st.sidebar.radio('Navigate to:',
    ['🏠 Home',
     '📋 Data Overview',
     '🏆 RFM Analysis',
     '💵 Payment Default Prediction',
     '📥 Download Results'])
st.sidebar.markdown('---')
st.sidebar.markdown('**Built by:** Joanna Mariam Varkey')
st.sidebar.markdown('**AdOnMo Summer Internship 2026**')

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_models():
    with open('lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return lr_model, scaler

lr_model, scaler = load_models()

# ================================
# LOAD CATEGORY MASTER
# ================================

# ================================
# LOAD CATEGORY MASTER
# ================================
@st.cache_data
def load_category_master():
    cat = pd.read_excel('Copy of Category Master - Joanna.xlsx')
    return cat.drop_duplicates(subset='CRM Brand Name')[
        ['CRM Brand Name','CRM Category','CRM Sub-Category','Type']]

cat_master = load_category_master()

# ================================
# FUZZY MATCHING FUNCTION
# ================================
from thefuzz import process, fuzz

def fuzzy_match_category(client_name, cat_master, threshold=80):
    """
    Matches a client name to the closest name in the 
    category master using fuzzy matching.
    Returns matched category info or None if below threshold.
    """
    # Clean the client name first
    client_clean = (str(client_name)
                    .strip()
                    .replace('\xa0', ' ')  # fix hidden characters
                    .replace('\u00a0', ' '))

    # Get all brand names from master
    brand_names = cat_master['CRM Brand Name'].tolist()

    # Find best match
    match = process.extractOne(
        client_clean,
        brand_names,
        scorer=fuzz.token_sort_ratio)

    if match and match[1] >= threshold:
        matched_name  = match[0]
        matched_score = match[1]
        matched_row   = cat_master[
            cat_master['CRM Brand Name'] == matched_name].iloc[0]
        return {
            'CRM Brand Name'    : matched_name,
            'CRM Category'      : matched_row['CRM Category'],
            'CRM Sub-Category'  : matched_row['CRM Sub-Category'],
            'Type'              : matched_row['Type'],
            'Match Score'       : matched_score
        }
    return None

def apply_fuzzy_matching(df, cat_master, threshold=80):
    """
    Applies fuzzy matching to all clients in the dataset
    and adds category columns
    """
    categories    = []
    sub_categories = []
    types         = []
    match_scores  = []

    for client in df['Client.Name']:
        # First try exact match
        exact = cat_master[
            cat_master['CRM Brand Name'] == client]

        if not exact.empty:
            row = exact.iloc[0]
            categories.append(row['CRM Category'])
            sub_categories.append(row['CRM Sub-Category'])
            types.append(row['Type'])
            match_scores.append(100)
        else:
            # Try fuzzy match
            result = fuzzy_match_category(
                client, cat_master, threshold)
            if result:
                categories.append(result['CRM Category'])
                sub_categories.append(result['CRM Sub-Category'])
                types.append(result['Type'])
                match_scores.append(result['Match Score'])
            else:
                categories.append(None)
                sub_categories.append(None)
                types.append(None)
                match_scores.append(0)

    df['Client_Category']     = categories
    df['Client_Sub_Category'] = sub_categories
    df['Client_Type']         = types
    df['Match_Score']         = match_scores

    return df

# ================================
# CLEANING FUNCTION
# ================================
def clean_data(df):
    # Rename columns
    col_map = {
        'Screen Type'                          : 'Screen.Type',
        'POC Name'                             : 'POC.Name',
        'POC Manager'                          : 'POC.Manager',
        'Client Name'                          : 'Client.Name',
        'Start Date'                           : 'Start.Date',
        'End Date'                             : 'End.Date',
        'No of Screens/Poster/Societies'       : 'No.of.Screens.Poster.Societies',
        'Campaign From'                        : 'Campaign.From',
        'Campaign Type'                        : 'Campaign.Type',
        'Selected Cities'                      : 'Selected.Cities',
        'Price Per Screen Per Month'           : 'Price.Per.Screen.Per.Month',
        'Order Value'                          : 'Order_Value_Clean',
        'Payment Status\n(Paid/Partially Paid)': 'Payment_Status_Raw',
        'Pending Amount'                       : 'Pending_Amount_Clean',
        'Campaign Duration'                    : 'Campaign.Duration',
        'Per Day Value'                        : 'Per.Day.Value',
    }
    # Rename monthly columns
    month_labels = ['Jan.25','Feb.25','Mar.25','Apr.25','May.25','Jun.25',
                    'Jul.25','Aug.25','Sep.25','Oct.25','Nov.25','Dec.25',
                    'Jan.26','Feb.26','Mar.26']
    datetime_cols = [c for c in df.columns if isinstance(c, datetime)]
    for i, col in enumerate(datetime_cols):
        if i < len(month_labels):
            col_map[col] = month_labels[i]

    df.rename(columns=col_map, inplace=True)

    # Fix Screen Type
    df['Screen.Type'] = 'Digital Screens'

    # Extract POC First Name
    def extract_poc(email):
        name = re.sub(r'@.*', '', str(email))
        name = re.sub(r'[._]', ' ', name)
        return name.title().strip()

    if 'POC.Name' in df.columns:
        df = df[df['POC.Name'].notna()]
        df = df[~df['POC.Name'].astype(str).str.match(r'^\d')]
        df['POC_First_Name'] = df['POC.Name'].apply(extract_poc)

    # Fix Client Names
    client_map = {
        'CaratLane'                               : 'Caratlane',
        'GoSwadeshi by Go coop'                   : 'GoSwadeshi by Gocoop',
        'Mytruso'                                 : 'MyTruso',
        'Kempegowda International Airport Bengaluru': 'Kempegowda International Airport',
        'Schools Easy'                            : 'SchoolsEasy',
        'Shibuya xing'                            : 'Shibuya Xing',
        'Swish Food Delivery'                     : 'Swish',
        'Vijayalakshmi Silk'                      : 'Vijayalakshmi Silks',
        'Prestige'                                : 'Prestige Group',
        'Prestige Constructions'                  : 'Prestige Group',
        'Swiggy Food Delivery'                    : 'Swiggy',
        'Amazon'                                  : 'Amazon Fashion',
    }
    if 'Client.Name' in df.columns:
        df['Client.Name'] = df['Client.Name'].replace(client_map)

    # Parse Dates
    df['Start.Date'] = pd.to_datetime(df['Start.Date'], errors='coerce')
    df['End.Date']   = pd.to_datetime(df['End.Date'],   errors='coerce')
    df['Start_Year']  = df['Start.Date'].dt.year
    df['Start_Month'] = df['Start.Date'].dt.strftime('%B')
    df['End_Year']    = df['End.Date'].dt.year
    df['End_Month']   = df['End.Date'].dt.strftime('%B')
    df['Campaign_Days'] = (df['End.Date'] - df['Start.Date']).dt.days.fillna(0).astype(int)
    df.loc[df['Campaign_Days'] == 0, 'Campaign_Days'] = 1
    df['Campaign_number_Months'] = ((df['End.Date'].dt.year - df['Start.Date'].dt.year) * 12 +
                                     (df['End.Date'].dt.month - df['Start.Date'].dt.month))

    # Fix Campaign.From
    from_map = {
        'Direct'            : 'Direct Client (Regional / hyper-local)',
        'Direct Hyperlocal' : 'Direct Client (Regional / hyper-local)',
        'Direct Regional'   : 'Direct Client (Regional / hyper-local)',
        'Direct Client (National)': 'Direct National',
    }
    if 'Campaign.From' in df.columns:
        df['Campaign.From'] = df['Campaign.From'].replace(from_map)

    # Fix Campaign Type
    if 'Campaign.Type' in df.columns:
        df['Campaign.Type'] = df['Campaign.Type'].str.strip().replace(
            'SIngle City', 'Single City')

    # Number of Cities
    if 'Selected.Cities' in df.columns:
        df['Number_of_Cities'] = df['Selected.Cities'].astype(str).str.count(',') + 1

    # Fix numeric columns
    # Fix Order Value
    if 'Order_Value_Clean' in df.columns:
        df['Order_Value_Clean'] = pd.to_numeric(
            df['Order_Value_Clean'].astype(str).str.replace(',',''),
            errors='coerce').fillna(0).astype(int)
    elif 'Order Value' in df.columns:
        df['Order_Value_Clean'] = pd.to_numeric(
            df['Order Value'].astype(str).str.replace(',',''),
            errors='coerce').fillna(0).astype(int)

    # Fix Pending Amount
    # NaN means unpaid — set pending = order value
    # Fix Pending Amount
    if 'Pending_Amount_Clean' in df.columns:
        df['Pending_Amount_Clean'] = pd.to_numeric(
            df['Pending_Amount_Clean'].astype(str).str.replace(',',''),
            errors='coerce')
    elif 'Pending Amount' in df.columns:
        df['Pending_Amount_Clean'] = pd.to_numeric(
            df['Pending Amount'].astype(str).str.replace(',',''),
            errors='coerce')
    else:
        df['Pending_Amount_Clean'] = 0

    # Smart fill — if Payment Status is Paid → pending = 0
    # If Payment Status is not Paid and pending is NaN → pending = order value
    if 'Payment_Status_Raw' not in df.columns and \
       'Payment Status\n(Paid/Partially Paid)' in df.columns:
        df['Payment_Status_Raw'] = df['Payment Status\n(Paid/Partially Paid)']

    def smart_fill_pending(row):
        pending = row['Pending_Amount_Clean']
        order   = row['Order_Value_Clean']
        status  = str(row.get('Payment_Status_Raw', '')).strip().upper()
        if pd.notna(pending):
            return pending
        elif status == 'PAID':
            return 0
        else:
            return order

    df['Pending_Amount_Clean'] = df.apply(
        smart_fill_pending, axis=1).fillna(0).astype(int)

    # Monthly columns
    month_cols = ['Jan.25','Feb.25','Mar.25','Apr.25','May.25','Jun.25',
                  'Jul.25','Aug.25','Sep.25','Oct.25','Nov.25','Dec.25',
                  'Jan.26','Feb.26','Mar.26']
    for col in month_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Payment Status
    def get_payment_status(row):
        pending = row.get('Pending_Amount_Clean', 0)
        order   = row.get('Order_Value_Clean', 0)
        status  = str(row.get('Payment_Status_Raw', '')).strip().upper()
        if status == 'PAID':
            return 'Paid'
        elif pending > 0 and pending < order:
            return 'Partially Paid'
        else:
            return 'Not Paid'

    # Map raw payment status column
    if 'Payment Status\n(Paid/Partially Paid)' in df.columns:
        df['Payment_Status_Raw'] = df['Payment Status\n(Paid/Partially Paid)']
    
    if 'Payment_Status_Raw' in df.columns:
        df['Payment_Status_Clean'] = df.apply(get_payment_status, axis=1)
    elif 'Payment_Status_Clean' not in df.columns:
        df['Payment_Status_Clean'] = 'Not Paid'

    # Price Per Screen
    def extract_price(val):
        prices = re.findall(r'-\s*([\d.]+)', str(val))
        clean  = []
        for p in prices:
            try:
                clean.append(float(p.rstrip('.')))
            except:
                continue
        return round(np.mean(clean), 2) if clean else np.nan

    if 'Price.Per.Screen.Per.Month' in df.columns:
        df['Price_Per_Screen_Avg'] = df['Price.Per.Screen.Per.Month'].apply(extract_price)
        df['Price_Per_Screen_Avg'] = df['Price_Per_Screen_Avg'].fillna(
            df['Price_Per_Screen_Avg'].median())
    elif 'Price_Per_Screen_Avg' not in df.columns:
        df['Price_Per_Screen_Avg'] = 0

    # Screen Tier
    def get_tier(val):
        val = str(val)
        if 'PLATINUM' in val: return 'PLATINUM'
        elif 'GOLD'   in val: return 'GOLD'
        elif 'SILVER' in val: return 'SILVER'
        return 'Unknown'

    if 'Price.Per.Screen.Per.Month' in df.columns:
        df['Screen_Tier'] = df['Price.Per.Screen.Per.Month'].apply(get_tier)
    elif 'Screen_Tier' not in df.columns:
        df['Screen_Tier'] = 'Unknown'

    # Fix Screens
    if 'No.of.Screens.Poster.Societies' in df.columns:
        df['No.of.Screens.Poster.Societies'] = pd.to_numeric(
            df['No.of.Screens.Poster.Societies'], errors='coerce').fillna(0).astype(int)

    # Flags
    df['Is_Zero_Value'] = (df['Order_Value_Clean'] == 0).astype(int)
    df['Collected']     = df['Order_Value_Clean'] - df['Pending_Amount_Clean']
    df['Collection_Rate'] = np.where(
        df['Order_Value_Clean'] > 0,
        (df['Collected'] / df['Order_Value_Clean'] * 100).round(2), 0)

    # Add Category Master
    df = apply_fuzzy_matching(df, cat_master, threshold=80)
    return df

# ================================
# RFM FUNCTION
# ================================
def calculate_rfm(df):
    reference_date = df['Start.Date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('Client.Name').agg(
        Recency   = ('Start.Date',        lambda x: (reference_date - x.max()).days),
        Frequency = ('Order_Value_Clean',  'count'),
        Monetary  = ('Order_Value_Clean',  'sum')
    ).reset_index()

    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'),
                              q=4, labels=[1,2,3,4]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'),
                              q=4, labels=[1,2,3,4]).astype(int)
    rfm['RFM_Score']   = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['Total_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    def segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        score   = row['Total_Score']
        if r >= 4 and f >= 3 and m >= 3: return '🏆 Champion'
        elif r >= 3 and f >= 3 and m >= 2: return '💛 Loyal'
        elif r >= 3 and f >= 2 and m >= 2: return '🌱 Potential'
        elif r <= 2 and f >= 3 and m >= 3: return '⚠️ At Risk'
        elif score <= 4: return '❌ Lost'
        else: return '💔 Needs Attention'

    rfm['Segment'] = rfm.apply(segment, axis=1)
    return rfm

# ================================
# DEFAULT PREDICTION FUNCTION
# ================================
def predict_default(df, rfm):
    features = ['No.of.Screens.Poster.Societies',
                'Campaign_Days',
                'Number_of_Cities',
                'Order_Value_Clean',
                'Price_Per_Screen_Avg',
                'R_Score', 'F_Score', 'M_Score',
                'Total_Score', 'Is_Zero_Value',
                'Campaign.From', 'Campaign.Type',
                'Screen_Tier']

    df_model = df.merge(rfm[['Client.Name','R_Score','F_Score',
                              'M_Score','Total_Score']],
                        on='Client.Name', how='left')

    for col in ['R_Score','F_Score','M_Score','Total_Score']:
        df_model[col] = df_model[col].fillna(df_model[col].median())

    le = LabelEncoder()
    for col in ['Campaign.From','Campaign.Type','Screen_Tier']:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model[features]
    df_model['Default_Probability'] = lr_model.predict_proba(X)[:, 1]
    df_model['Risk_Category'] = pd.cut(
        df_model['Default_Probability'],
        bins=[0, 0.4, 0.6, 0.8, 1.0],
        labels=['🟢 Low Risk','🟡 Medium Risk',
                '🟠 High Risk','🔴 Very High Risk'])
    return df_model

# ================================
# DOWNLOAD FUNCTION
# ================================
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# ================================
# PAGE 1 — HOME
# ================================
if page == '🏠 Home':
    st.image('adonmo_2nd_image.webp', width=250)
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="
            color: #1B2A4A;
            font-size: 72px;
            font-weight: 900;
            border-bottom: 4px solid #00B4D8;
            padding-bottom: 16px;
            letter-spacing: 3px;
            margin-bottom: 8px;
            line-height: 1.1;
            text-transform: uppercase;
            font-family: 'Inter', sans-serif;
            ">
            AdOnMo RWA Intelligence Platform
        </div>
        <div style="
            color: #00B4D8;
            font-size: 22px;
            font-weight: 700;
            margin: 8px 0 0 0;
            letter-spacing: 3px;
            text-transform: uppercase;
            font-family: 'Inter', sans-serif;
            ">
            A CLIENT ANALYSIS & RISK MANAGEMENT SYSTEM
        </div>
        <div style="
            color: #64748B;
            font-size: 15px;
            font-weight: 500;
            margin: 12px 0 0 0;
            letter-spacing: 1px;
            font-family: 'Inter', sans-serif;
            ">
             Powered by Machine Learning &nbsp;|&nbsp; 
             Built for AdOnMo Sales Team &nbsp;|&nbsp; 
             RWA Campaign Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('---')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background:#EFF6FF; border-left:4px solid #00B4D8; 
        border-radius:10px; padding:24px;
        transition: all 0.3s ease;">
        <h4 style="color:#1B2A4A; margin:0 0 12px 0;
        font-size:20px; font-weight:800; letter-spacing:0.5px;">
        🏆 RFM Analysis</h4>
        <p style="color:#374151; margin:0; font-size:15px; 
        font-weight:600; line-height:1.6;">
        Segment clients into Champions, Loyal, At Risk and Lost 
        based on their campaign behaviour</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#FFF7ED; border-left:4px solid #F59E0B; 
        border-radius:10px; padding:24px;
        transition: all 0.3s ease;">
        <h4 style="color:#1B2A4A; margin:0 0 12px 0;
        font-size:20px; font-weight:800; letter-spacing:0.5px;">
        💵 Payment Default Prediction</h4>
        <p style="color:#374151; margin:0; font-size:15px; 
        font-weight:600; line-height:1.6;">
        Identify high risk campaigns before execution 
        using Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background:#F0FDF4; border-left:4px solid #22C55E; 
        border-radius:10px; padding:24px;
        transition: all 0.3s ease;">
        <h4 style="color:#1B2A4A; margin:0 0 12px 0;
        font-size:20px; font-weight:800; letter-spacing:0.5px;">
        📥 Download Results</h4>
        <p style="color:#374151; margin:0; font-size:15px; 
        font-weight:600; line-height:1.6;">
        Export all findings as Excel for the sales team 
        to action immediately</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('---')
    styled_heading('Upload Your RWA Dataset')
    st.markdown('Upload either the **cleaned** or **uncleaned** RWA Excel file the app will handle everything automatically!')

    uploaded_file = st.file_uploader(
        'Choose your RWA Excel file',
        type=['xlsx','csv'],
        help='Upload the RWA dataset — cleaned or uncleaned both work!')

    if uploaded_file is not None:
        with st.spinner('Loading and cleaning your data...'):
            try:
                if uploaded_file.name.endswith('.csv'):
                    raw_df = pd.read_csv(uploaded_file)
                else:
                    raw_df = pd.read_excel(uploaded_file)

                # Check if cleaned or uncleaned
                if 'Client.Name' in raw_df.columns:
                    df = raw_df.copy()
                    df['Start.Date'] = pd.to_datetime(df['Start.Date'], errors='coerce')
                    df['End.Date']   = pd.to_datetime(df['End.Date'],   errors='coerce')
                    df['Collected']  = df['Order_Value_Clean'] - df['Pending_Amount_Clean']
                    df['Collection_Rate'] = np.where(
                        df['Order_Value_Clean'] > 0,
                        (df['Collected'] / df['Order_Value_Clean'] * 100).round(2), 0)
                    st.session_state['df'] = df
                else:
                    df = clean_data(raw_df)
                    st.session_state['df'] = df

                st.success(f'Dataset loaded successfully! {len(df)} campaigns found!')
                styled_heading('Quick Summary')
                col1, col2, col3, col4 = st.columns(4)
                col1.metric('Total Campaigns', len(df))
                col2.metric('Total Clients', df['Client.Name'].nunique())
                col3.metric('Total Order Value', f'₹{df["Order_Value_Clean"].sum()/1e7:.2f} Cr')
                col4.metric('Collection Rate', f'{df["Collection_Rate"].mean():.1f}%')

                st.info('Use the sidebar to navigate to different sections!')

            except Exception as e:
                st.error(f'Error loading file: {str(e)}')
                st.info('Please make sure your file is a valid RWA Excel or CSV file!')
    else:
        st.markdown('---')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <h3 style="color:#1B2A4A; font-size:22px; 
            font-weight:700; margin-bottom:16px;">
            ℹ️ How to use this app:</h3>
            <p style="color:#1B2A4A; font-size:18px; 
            font-weight:600; margin:8px 0;">
            1. Upload your RWA Excel file above</p>
            <p style="color:#1B2A4A; font-size:18px; 
            font-weight:600; margin:8px 0;">
            2. The app will automatically clean and process the data</p>
            <p style="color:#1B2A4A; font-size:18px; 
            font-weight:600; margin:8px 0;">
            3. Navigate using the sidebar to explore:</p>
            <p style="color:#00B4D8; font-size:16px; 
            font-weight:600; margin:4px 0 4px 20px;">
            📋 Data Overview</p>
            <p style="color:#00B4D8; font-size:16px; 
            font-weight:600; margin:4px 0 4px 20px;">
            🏆 RFM Analysis</p>
            <p style="color:#00B4D8; font-size:16px; 
            font-weight:600; margin:4px 0 4px 20px;">
            💵 Payment Default Prediction</p>
            <p style="color:#00B4D8; font-size:16px; 
            font-weight:600; margin:4px 0 4px 20px;">
            📥 Download Results</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <h3 style="color:#1B2A4A; font-size:22px; 
            font-weight:700; margin-bottom:16px;">
            ℹ️ About this app:</h3>
            <p style="color:#1B2A4A; font-size:18px; 
            font-weight:600; margin:8px 0;">
            Built for <span style="color:#00B4D8;">
            AdOnMo Sales Team</span></p>
            <p style="color:#1B2A4A; font-size:18px; 
            font-weight:600; margin:8px 0;">
            No coding knowledge required</p>
            <p style="color:#1B2A4A; font-size:18px; 
            font-weight:600; margin:8px 0;">
            Works with raw or cleaned RWA data</p>
            <p style="color:#1B2A4A; font-size:18px; 
            font-weight:600; margin:8px 0;">
            Powered by Machine Learning</p>
            <p style="color:#1B2A4A; font-size:18px; 
            font-weight:600; margin:8px 0;">
            Results downloadable as Excel</p>
            </div>
            """, unsafe_allow_html=True)
            
# to add the data_overview.py page after the home page

# ================================
# PAGE 2 — DATA OVERVIEW
# ================================
elif page == '📋 Data Overview':
    st.title('📋 Data Overview')
    st.markdown('---')

    if 'df' not in st.session_state:
        st.warning('⚠️ Please upload your RWA dataset on the Home page first!')
        st.stop()

    df = st.session_state['df']

    # Key Metrics
    styled_heading('Key Metrics')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric('Total Campaigns',   f'{len(df):,}')
    col2.metric('Total Clients',     f'{df["Client.Name"].nunique():,}')
    col3.metric('Total Order Value', f'₹{df["Order_Value_Clean"].sum()/1e7:.2f} Cr')
    col4.metric('Total Collected',   f'₹{df["Collected"].sum()/1e7:.2f} Cr')
    col5.metric('Total Pending',     f'₹{df["Pending_Amount_Clean"].sum()/1e7:.2f} Cr')

    st.markdown('---')

    # Payment Status & Campaign Type
    styled_heading('📈 Campaign Distribution')
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        pay_counts = df['Payment_Status_Clean'].value_counts()
        colors = ['#e74c3c','#2ecc71','#f39c12']
        ax.pie(pay_counts, labels=pay_counts.index,
               autopct='%1.1f%%', colors=colors,
               startangle=90, textprops={'fontsize': 9})
        ax.set_title('Payment Status Distribution',
                     fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        camp_counts = df['Campaign.Type'].value_counts()
        sns.barplot(ax=ax, x=camp_counts.index,
                    y=camp_counts.values,
                    palette=['steelblue','coral'])
        for i, v in enumerate(camp_counts.values):
            ax.text(i, v + 1, str(v),
                    ha='center', fontweight='bold')
        ax.set_title('Campaign Type Distribution',
                     fontweight='bold')
        ax.set_xlabel('Campaign Type')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close()

    st.markdown('---')

    # Monthly Revenue Trend
    styled_heading('Monthly Revenue Trend')
    REV_COLS = ['Jan.25.Rev','Feb.25.Rev','Mar.25.Rev','Apr.25.Rev',
                'May.25.Rev','Jun.25.Rev','Jul.25.Rev','Aug.25.Rev',
                'Sep.25.Rev','Oct.25.Rev','Nov.25.Rev','Dec.25.Rev',
                'Jan.26.Rev','Feb.26.Rev','Mar.26.Rev']
    REV_COLS_ALT = ['Jan-25 Rev','Feb-25 Rev','Mar-25 Rev','Apr-25 Rev',
                    'May-25 Rev','Jun-25 Rev','Jul-25 Rev','Aug-25 Rev',
                    'Sep-25 Rev','Oct-25 Rev','Nov-25 Rev','Dec-25 Rev',
                    'Jan-26 Rev','Feb-26 Rev','Mar-26 Rev']
    MONTH_LABELS = ['Jan-25','Feb-25','Mar-25','Apr-25','May-25',
                    'Jun-25','Jul-25','Aug-25','Sep-25','Oct-25',
                    'Nov-25','Dec-25','Jan-26','Feb-26','Mar-26']

    # Check which format is available
    available_rev = [c for c in REV_COLS if c in df.columns]
    available_labels = [MONTH_LABELS[i] for i, c in enumerate(REV_COLS)
                        if c in df.columns]

    # If standard format not found try alternate format
    if not available_rev:
        available_rev = [c for c in REV_COLS_ALT if c in df.columns]
        available_labels = [MONTH_LABELS[i] for i, c in enumerate(REV_COLS_ALT)
                            if c in df.columns]

    if available_rev:
        monthly_rev = df[available_rev].sum()
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(available_labels, monthly_rev.values/1e5,
                marker='o', color='steelblue', linewidth=2.5)
        ax.fill_between(available_labels,
                        monthly_rev.values/1e5,
                        alpha=0.2, color='steelblue')
        for i, val in enumerate(monthly_rev.values/1e5):
            ax.annotate(f'₹{val:.1f}L',
                        (available_labels[i], val),
                        textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=7)
        ax.set_title('Monthly Revenue Trend',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Revenue (₹ Lakhs)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info('Monthly revenue columns not found in this dataset')

    st.markdown('---')

    # Top 10 Clients
    styled_heading('Top 10 Clients by Order Value')
    top_clients = (df.groupby('Client.Name')['Order_Value_Clean']
                   .sum().sort_values(ascending=False).head(10)
                   .reset_index())
    top_clients.columns = ['Client', 'Total Order Value (₹)']
    top_clients['Total Order Value (₹)'] = top_clients['Total Order Value (₹)'].apply(
        lambda x: f'₹{x:,.0f}')
    st.dataframe(top_clients, use_container_width=True)

    st.markdown('---')

    # Campaign Source Performance
    styled_heading('Campaign Source Performance')
    source_perf = (df.groupby('Campaign.From')
                   .agg(Total_Revenue  = ('Order_Value_Clean', 'sum'),
                        Campaign_Count = ('Order_Value_Clean', 'count'),
                        Avg_Deal_Size  = ('Order_Value_Clean', 'mean'),
                        Avg_Collection = ('Collection_Rate',   'mean'))
                   .sort_values('Total_Revenue', ascending=False)
                   .reset_index())
    source_perf['Total_Revenue']  = source_perf['Total_Revenue'].apply(lambda x: f'₹{x:,.0f}')
    source_perf['Avg_Deal_Size']  = source_perf['Avg_Deal_Size'].apply(lambda x: f'₹{x:,.0f}')
    source_perf['Avg_Collection'] = source_perf['Avg_Collection'].apply(lambda x: f'{x:.1f}%')
    source_perf.columns = ['Campaign Source','Total Revenue',
                           'Campaign Count','Avg Deal Size','Avg Collection Rate']
    st.dataframe(source_perf, use_container_width=True)

    st.markdown('---')

    # Raw Data Preview
    styled_heading('Raw Data Preview')
    st.dataframe(df.head(20), use_container_width=True)

# ================================
# PAGE 3 — RFM ANALYSIS
# ================================
elif page == '🏆 RFM Analysis':
    st.title('🏆 RFM Analysis')
    st.markdown('---')

    if 'df' not in st.session_state:
        st.warning('⚠️ Please upload your RWA dataset on the Home page first!')
        st.stop()

    df = st.session_state['df']

    with st.spinner('Running RFM Analysis...'):
        rfm = calculate_rfm(df)
        st.session_state['rfm'] = rfm

    st.success(f'RFM Analysis complete! {len(rfm)} clients segmented!')

    # Segment Metrics
    styled_heading('Segment Overview')
    seg_counts = rfm['Segment'].value_counts()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    cols = [col1, col2, col3, col4, col5, col6]
    segments = ['🏆 Champion', '💛 Loyal', '🌱 Potential',
                '💔 Needs Attention', '⚠️ At Risk', '❌ Lost']
    for i, seg in enumerate(segments):
        count = seg_counts.get(seg, 0)
        cols[i].metric(seg, count)

    st.markdown('---')

    # Segment Distribution Charts
    styled_heading('Segment Distribution')
    col1, col2 = st.columns(2)

    SEGMENT_COLORS = {
        '🏆 Champion'       : '#2ecc71',
        '💛 Loyal'          : '#3498db',
        '🌱 Potential'      : '#9b59b6',
        '💔 Needs Attention': '#f39c12',
        '⚠️ At Risk'        : '#e67e22',
        '❌ Lost'           : '#e74c3c'
    }

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = [SEGMENT_COLORS.get(s, '#gray') for s in seg_counts.index]
        ax.pie(seg_counts, labels=seg_counts.index,
               autopct='%1.1f%%', colors=colors,
               startangle=90, textprops={'fontsize': 8})
        ax.set_title('Client Segment Distribution',
                     fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        seg_monetary = rfm.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        colors_bar = [SEGMENT_COLORS.get(s, 'gray') for s in seg_monetary.index]
        bars = ax.barh(seg_monetary.index,
                       seg_monetary.values/1e5,
                       color=colors_bar)
        for bar in bars:
            ax.text(bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height()/2,
                    f'₹{bar.get_width():.1f}L',
                    va='center', fontsize=8)
        ax.set_title('Avg Spend per Segment',
                     fontweight='bold')
        ax.set_xlabel('Avg Monetary Value (₹ Lakhs)')
        st.pyplot(fig)
        plt.close()

    st.markdown('---')

    # RFM Scatter Plot
    styled_heading('RFM Scatter Plot')
    fig, ax = plt.subplots(figsize=(12, 6))
    for segment, color in SEGMENT_COLORS.items():
        mask = rfm['Segment'] == segment
        ax.scatter(rfm[mask]['Recency'],
                   rfm[mask]['Monetary']/1e5,
                   c=color, label=segment,
                   s=rfm[mask]['Frequency']*20,
                   alpha=0.7, edgecolors='black',
                   linewidth=0.5)
    ax.set_title('RFM Scatter Plot (Bubble size = Frequency)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Recency (Days) → Lower is Better')
    ax.set_ylabel('Monetary Value (₹ Lakhs) → Higher is Better')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('---')

    # Segment Filter
    styled_heading('Explore by Segment')
    selected_segment = st.selectbox(
        'Select a segment to explore:',
        ['All'] + segments)

    if selected_segment == 'All':
        filtered_rfm = rfm
    else:
        filtered_rfm = rfm[rfm['Segment'] == selected_segment]

    st.dataframe(filtered_rfm[['Client.Name','Recency','Frequency',
                                'Monetary','R_Score','F_Score',
                                'M_Score','Total_Score','Segment']]
                 .sort_values('Monetary', ascending=False),
                 use_container_width=True)

    st.markdown('---')

    # Champion Clients
    styled_heading('Top Champion Clients')
    champions = rfm[rfm['Segment'] == '🏆 Champion'].sort_values(
        'Monetary', ascending=False).head(10)
    if len(champions) > 0:
        for _, row in champions.iterrows():
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f"**{row['Client.Name']}**")
            col2.markdown(f"📅 Last seen: **{row['Recency']} days ago**")
            col3.markdown(f"🔁 Campaigns: **{row['Frequency']}**")
            col4.markdown(f"💰 Total Spend: **₹{row['Monetary']/1e5:.1f}L**")
    else:
        st.info('No Champion clients found in this dataset')

    st.markdown('---')

    # Segment Summary Table
    styled_heading('📋 Segment Summary')
    seg_summary = rfm.groupby('Segment').agg(
        Client_Count  = ('Client.Name', 'count'),
        Avg_Recency   = ('Recency',     'mean'),
        Avg_Frequency = ('Frequency',   'mean'),
        Avg_Monetary  = ('Monetary',    'mean')
    ).round(2).reset_index()
    seg_summary['Avg_Monetary'] = seg_summary['Avg_Monetary'].apply(
        lambda x: f'₹{x:,.0f}')
    st.dataframe(seg_summary, use_container_width=True)

# ================================
# PAGE 4 — PAYMENT DEFAULT PREDICTION
# ================================
elif page == '💵 Payment Default Prediction':
    st.title('💵 Payment Default Prediction')
    st.markdown('---')

    if 'df' not in st.session_state:
        st.warning('⚠️ Please upload your RWA dataset on the Home page first!')
        st.stop()

    df = st.session_state['df']

    # Run RFM if not already done
    if 'rfm' not in st.session_state:
        with st.spinner('Running RFM Analysis first...'):
            rfm = calculate_rfm(df)
            st.session_state['rfm'] = rfm
    else:
        rfm = st.session_state['rfm']

    # Run Default Prediction
    with st.spinner('Running Payment Default Prediction...'):
        df_model = predict_default(df, rfm)
        st.session_state['df_model'] = df_model

    st.success('Payment Default Prediction complete!')

    st.markdown('---')

    # Risk Distribution Metrics
    styled_heading('⚠️ Risk Distribution')
    risk_counts = df_model['Risk_Category'].value_counts()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('🔴 Very High Risk',
                risk_counts.get('🔴 Very High Risk', 0),
                delta=f"{risk_counts.get('🔴 Very High Risk', 0)/len(df_model)*100:.1f}%",
                delta_color='inverse')
    col2.metric('🟠 High Risk',
                risk_counts.get('🟠 High Risk', 0),
                delta=f"{risk_counts.get('🟠 High Risk', 0)/len(df_model)*100:.1f}%",
                delta_color='inverse')
    col3.metric('🟡 Medium Risk',
                risk_counts.get('🟡 Medium Risk', 0),
                delta=f"{risk_counts.get('🟡 Medium Risk', 0)/len(df_model)*100:.1f}%",
                delta_color='off')
    col4.metric('🟢 Low Risk',
                risk_counts.get('🟢 Low Risk', 0),
                delta=f"{risk_counts.get('🟢 Low Risk', 0)/len(df_model)*100:.1f}%",
                delta_color='normal')

    st.markdown('---')

    # Risk Distribution Chart
    styled_heading('Risk Category Distribution')
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        risk_colors = ['#e74c3c','#e67e22','#f39c12','#2ecc71']
        risk_order  = ['🔴 Very High Risk','🟠 High Risk',
                       '🟡 Medium Risk','🟢 Low Risk']
        risk_vals   = [risk_counts.get(r, 0) for r in risk_order]
        ax.pie(risk_vals, labels=risk_order,
               autopct='%1.1f%%', colors=risk_colors,
               startangle=90, textprops={'fontsize': 8})
        ax.set_title('Risk Category Distribution',
                     fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        risk_revenue = df_model.groupby('Risk_Category')['Order_Value_Clean'].sum()
        risk_revenue = risk_revenue.reindex(risk_order, fill_value=0)
        bars = ax.barh(risk_order, risk_revenue.values/1e6,
                       color=risk_colors)
        for bar in bars:
            ax.text(bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height()/2,
                    f'₹{bar.get_width():.1f}M',
                    va='center', fontsize=8)
        ax.set_title('Revenue at Risk by Category',
                     fontweight='bold')
        ax.set_xlabel('Total Order Value (₹ Million)')
        st.pyplot(fig)
        plt.close()

    st.markdown('---')

    # Risk Filter
    styled_heading('Explore by Risk Category')
    selected_risk = st.selectbox(
        'Select a risk category to explore:',
        ['All','🔴 Very High Risk','🟠 High Risk',
         '🟡 Medium Risk','🟢 Low Risk'])

    if selected_risk == 'All':
        filtered = df_model
    else:
        filtered = df_model[df_model['Risk_Category'] == selected_risk]

    st.markdown(f'**{len(filtered)} campaigns found**')
    st.dataframe(
        filtered[['Client.Name','Order_Value_Clean',
                  'Campaign_Days','Payment_Status_Clean',
                  'Default_Probability','Risk_Category']]
        .sort_values('Default_Probability', ascending=False)
        .reset_index(drop=True),
        use_container_width=True)

    st.markdown('---')

    # High Risk Client Summary
    styled_heading('🚨 High Priority Action List')
    st.markdown('These clients require **immediate follow up** by the sales team:')

    high_risk = (df_model[df_model['Risk_Category'].isin(
        ['🔴 Very High Risk','🟠 High Risk'])]
        .groupby('Client.Name')
        .agg(Campaign_Count    = ('Order_Value_Clean', 'count'),
             Total_Value       = ('Order_Value_Clean', 'sum'),
             Avg_Probability   = ('Default_Probability', 'mean'))
        .sort_values('Total_Value', ascending=False)
        .reset_index())

    high_risk['Total_Value']     = high_risk['Total_Value'].apply(
        lambda x: f'₹{x:,.0f}')
    high_risk['Avg_Probability'] = high_risk['Avg_Probability'].apply(
        lambda x: f'{x*100:.1f}%')
    high_risk.columns = ['Client','Campaign Count',
                         'Total Value at Risk','Avg Default Probability']
    st.dataframe(high_risk, use_container_width=True)

    st.markdown('---')

    # Model Performance
    styled_heading('Model Performance Summary')
    col1, col2, col3 = st.columns(3)
    col1.metric('Model Used',     'Random Forest')
    col2.metric('Model Accuracy', '80.10%')
    col3.metric('ROC-AUC Score',  '0.7975')

    st.info('''
    📌 **How to interpret Default Probability:**
    - **0.00 — 0.40** → 🟢 Low Risk — Very likely to pay
    - **0.40 — 0.60** → 🟡 Medium Risk — Could go either way
    - **0.60 — 0.80** → 🟠 High Risk — Likely to default
    - **0.80 — 1.00** → 🔴 Very High Risk — Almost certain to default
    ''')

# ================================
# PAGE 5 — DOWNLOAD RESULTS
# ================================
elif page == '📥 Download Results':
    st.title('📥 Download Results')
    st.markdown('---')

    if 'df' not in st.session_state:
        st.warning('⚠️ Please upload your RWA dataset on the Home page first!')
        st.stop()

    df  = st.session_state['df']

    # Run RFM if not done
    if 'rfm' not in st.session_state:
        with st.spinner('Running RFM Analysis...'):
            rfm = calculate_rfm(df)
            st.session_state['rfm'] = rfm
    else:
        rfm = st.session_state['rfm']

    # Run Default Prediction if not done
    if 'df_model' not in st.session_state:
        with st.spinner('Running Payment Default Prediction...'):
            df_model = predict_default(df, rfm)
            st.session_state['df_model'] = df_model
    else:
        df_model = st.session_state['df_model']

    st.success('All analysis complete — ready to download!')
    st.markdown('---')

    # Download Option 1 — RFM Results
    styled_heading('Download RFM Segmentation Results')
    st.markdown('Contains client segments, RFM scores and monetary values')
    rfm_download = rfm[['Client.Name','Recency','Frequency',
                         'Monetary','R_Score','F_Score','M_Score',
                         'Total_Score','Segment']].copy()
    rfm_download['Monetary'] = rfm_download['Monetary'].apply(
        lambda x: f'₹{x:,.0f}')
    st.dataframe(rfm_download, use_container_width=True)
    st.download_button(
        label     = '📥 Download RFM Results as Excel',
        data      = to_excel(rfm_download),
        file_name = 'RWA_RFM_Results.xlsx',
        mime      = 'application/vnd.ms-excel')

    st.markdown('---')

    # Download Option 2 — Default Predictions
    styled_heading('Download Payment Default Predictions')
    st.markdown('Contains risk category and default probability for every campaign')
    pred_download = df_model[['Client.Name','Order_Value_Clean',
                               'Campaign_Days','Payment_Status_Clean',
                               'Default_Probability',
                               'Risk_Category']].copy()
    pred_download['Default_Probability'] = pred_download[
        'Default_Probability'].apply(lambda x: f'{x*100:.1f}%')
    pred_download['Order_Value_Clean'] = pred_download[
        'Order_Value_Clean'].apply(lambda x: f'₹{x:,.0f}')
    pred_download.columns = ['Client','Order Value','Campaign Days',
                              'Payment Status','Default Probability',
                              'Risk Category']
    st.dataframe(pred_download, use_container_width=True)
    st.download_button(
        label     = '📥 Download Default Predictions as Excel',
        data      = to_excel(pred_download),
        file_name = 'RWA_Default_Predictions.xlsx',
        mime      = 'application/vnd.ms-excel')

    st.markdown('---')

    # Download Option 3 — Combined Report
    styled_heading('Download Combined Report')
    st.markdown('Contains everything — RFM scores + Risk categories in one file')

    # Merge RFM and predictions
    combined = df_model[['Client.Name','Order_Value_Clean',
                          'Campaign_Days','Payment_Status_Clean',
                          'Default_Probability','Risk_Category']].copy()
    combined = combined.merge(
        rfm[['Client.Name','Recency','Frequency',
             'Monetary','Segment']],
        on='Client.Name', how='left')

    combined['Default_Probability'] = combined[
        'Default_Probability'].apply(lambda x: f'{x*100:.1f}%')
    combined['Order_Value_Clean'] = combined[
        'Order_Value_Clean'].apply(lambda x: f'₹{x:,.0f}')
    combined['Monetary'] = combined['Monetary'].apply(
        lambda x: f'₹{x:,.0f}' if pd.notna(x) else 'N/A')

    combined.columns = ['Client','Order Value','Campaign Days',
                        'Payment Status','Default Probability',
                        'Risk Category','Recency','Frequency',
                        'Total Spend','RFM Segment']

    st.dataframe(combined, use_container_width=True)
    st.download_button(
        label     = '📥 Download Combined Report as Excel',
        data      = to_excel(combined),
        file_name = 'RWA_Combined_Report.xlsx',
        mime      = 'application/vnd.ms-excel')

    st.markdown('---')

    # Summary Stats
    styled_heading(' Analysis Summary')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Campaigns',   f'{len(df):,}')
    col2.metric('Total Clients',     f'{len(rfm):,}')
    col3.metric('Champion Clients',  
                f'{len(rfm[rfm["Segment"]=="🏆 Champion"]):,}')
    col4.metric('Very High Risk',    
                f'{len(df_model[df_model["Risk_Category"]=="🔴 Very High Risk"]):,}')

    st.markdown('---')
    st.info('''
     **How to use these reports:**
    - **RFM Results** → Share with sales team to prioritize client outreach
    - **Default Predictions** → Share with finance team to flag risky campaigns
    - **Combined Report** → Share with management for complete overview
    ''')
    st.markdown('*Built by Joanna Varkey | AdOnMo Summer Internship 2026*')
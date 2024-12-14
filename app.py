import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(page_title="Chip Sales Dashboard", layout="wide")
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df['Tarix'] = pd.to_datetime(df['Tarix'])
        
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

df = load_data(r"C:\Users\gunel\Downloads\satis_streamlitapp\p\cleaned_data (2).csv")

st.title("🥔 Chip Sales Analytics Dashboard")
st.sidebar.title("Dashboard Filters")
date_range = st.sidebar.date_input("Select Date Range", 
                                   [df['Tarix'].min().date(), 
                                    df['Tarix'].max().date()])

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

filtered_df = df[(df['Tarix'] >= start_date) & (df['Tarix'] <= end_date)]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Sales Volume", f"{filtered_df['Məhsul sayi'].sum():.0f} units")
with col2:
    st.metric("Total Revenue", f"₼ {filtered_df['Ümumi satış'].sum():.2f}")
with col3:
    st.metric("Unique Stores", filtered_df['Mağaza'].nunique())

tab1, tab2, tab3, tab4 = st.tabs([
    "Chip Brands Analysis", 
    "Sales Trends", 
    "Store Performance", 
    "Product Details"
])

with tab1:
    st.header("Chip Brands Performance")
    
    brand_sales = filtered_df.groupby('chip_name').agg({
        'Ümumi satış': 'sum',
        'Məhsul sayi': 'sum'
    }).reset_index()
    brand_sales = brand_sales.sort_values('Ümumi satış', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_brand_sales = px.bar(brand_sales, 
                                 x='chip_name', 
                                 y='Ümumi satış', 
                                 title='Sales Volume by Chip Brand',
                                 labels={'chip_name': 'Chip Brand', 'Ümumi satış': 'Total Sales (₼)'},
                                 text='Ümumi satış')
        st.plotly_chart(fig_brand_sales)
    
    with col2:
        fig_brand_units = px.bar(brand_sales, 
                                 x='chip_name', 
                                 y='Məhsul sayi', 
                                 title='Units Sold by Chip Brand',
                                 labels={'chip_name': 'Chip Brand', 'Məhsul sayi': 'Total Units'},
                                 text='Məhsul sayi')
        st.plotly_chart(fig_brand_units)

with tab2:
    st.header("Sales Trends")
    daily_sales = filtered_df.groupby('Tarix').agg({
        'Ümumi satış': 'sum',
        'Məhsul sayi': 'sum'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_daily_sales = px.line(daily_sales, 
                                  x='Tarix', 
                                  y='Ümumi satış', 
                                  title='Daily Sales Trend (₼)',
                                  labels={'Tarix': 'Date', 'Ümumi satış': 'Total Sales (₼)'})
        st.plotly_chart(fig_daily_sales)
    
    with col2:
        fig_daily_units = px.line(daily_sales, 
                                  x='Tarix', 
                                  y='Məhsul sayi', 
                                  title='Daily Units Sold',
                                  labels={'Tarix': 'Date', 'Məhsul sayi': 'Total Units'})
        st.plotly_chart(fig_daily_units)

with tab3:
    st.header("Store Performance")
    store_sales = filtered_df.groupby('Mağaza').agg({
        'Ümumi satış': ['sum', 'mean'],
        'Məhsul sayi': 'sum'
    }).reset_index()
    store_sales.columns = ['Store', 'Total Sales', 'Avg Transaction Value', 'Total Units']
    store_sales = store_sales.sort_values('Total Sales', ascending=False)
    fig_store_sales = px.bar(store_sales, 
                              x='Store', 
                              y='Total Sales', 
                              title='Sales Performance by Store',
                              labels={'Store': 'Store', 'Total Sales': 'Total Sales (₼)'},
                              text='Total Sales')
    st.plotly_chart(fig_store_sales)

with tab4:
    st.header("Flavor and Weight Analysis")
    flavor_sales = filtered_df.groupby('flavor').agg({
        'Ümumi satış': 'sum',
        'Məhsul sayi': 'sum'
    }).reset_index()
    flavor_sales = flavor_sales.sort_values('Ümumi satış', ascending=False)
    weight_sales = filtered_df.groupby('weight').agg({
        'Ümumi satış': 'sum',
        'Məhsul sayi': 'sum'
    }).reset_index()
    weight_sales = weight_sales.sort_values('Ümumi satış', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_flavor_sales = px.pie(flavor_sales, 
                                  values='Ümumi satış', 
                                  names='flavor', 
                                  title='Sales Distribution by Flavor')
        st.plotly_chart(fig_flavor_sales)
    
    with col2:
        fig_weight_sales = px.bar(weight_sales, 
                                  x='weight', 
                                  y='Ümumi satış', 
                                  title='Sales by Chip Weight',
                                  labels={'weight': 'Chip Weight (g)', 'Ümumi satış': 'Total Sales (₼)'})
        st.plotly_chart(fig_weight_sales)
st.sidebar.header("Data Insights")
st.sidebar.write("Missing Values:")
st.sidebar.write(df.isnull().sum())

if st.sidebar.checkbox("View Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(filtered_df)

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the existing code from the previous script
df = load_data(r"C:\Users\gunel\Downloads\satis_streamlitapp\p\cleaned_data (2).csv")

# Load the best saved model
try:
    best_model = joblib.load("best_model.pkl")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Preprocessing function (similar to what was used during model training)
def create_preprocessor(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    
    # Remove 'Ümumi satış' from features if present
    if 'Ümumi satış' in numeric_features:
        numeric_features.remove('Ümumi satış')
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, numeric_features, categorical_features

# Create preprocessor
preprocessor, numeric_features, categorical_features = create_preprocessor(df.drop('Ümumi satış', axis=1))

# Fit the preprocessor on the entire dataset
X = df.drop('Ümumi satış', axis=1)
preprocessor.fit(X)

# Add a new tab for Sales Prediction
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Chip Brands Analysis", 
    "Sales Trends", 
    "Store Performance", 
    "Product Details",
    "Sales Prediction" # New tab
])

# ... (keep all previous tabs as they were)

# New tab for Sales Prediction
with tab5:
    st.header("Chip Sales Prediction")
    
    # Create input fields dynamically based on available features
    st.subheader("Enter Chip Sale Details for Prediction")
    
    # Prepare input containers
    input_data = {}
    
    # Numeric features
    st.write("Numeric Features:")
    numeric_cols = st.columns(len(numeric_features))
    for i, feature in enumerate(numeric_features):
        with numeric_cols[i]:
            input_data[feature] = st.number_input(feature, min_value=0.0, step=0.1)
    
    # Categorical features
    st.write("Categorical Features:")
    cat_cols = st.columns(len(categorical_features))
    for i, feature in enumerate(categorical_features):
        with cat_cols[i]:
            # Get unique values for each categorical feature
            unique_values = df[feature].unique()
            input_data[feature] = st.selectbox(feature, unique_values)
    
    # Prediction button
    if st.button("Predict Sales"):
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        try:
            # Preprocess the input
            input_transformed = preprocessor.transform(input_df)
            
            # Make prediction
            prediction = best_model.predict(input_transformed)
            
            # Display prediction
            st.success(f"Predicted Sales: ₼ {prediction[0]:.2f}")
            
            # Confidence visualization
            st.write("Prediction Confidence Visualization")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction[0],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Predicted Sales"},
                gauge = {
                    'axis': {'range': [0, df['Ümumi satış'].max()]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, df['Ümumi satış'].mean()], 'color': "lightblue"},
                        {'range': [df['Ümumi satış'].mean(), df['Ümumi satış'].max()], 'color': "blue"}
                    ]
                }
            ))
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Keep the rest of the original dashboard code
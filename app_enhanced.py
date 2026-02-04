import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


DATA_PATH = "Dataset/Features.csv"

EMISSION_MODEL_PATH = "Models_Trained/Emission_Sklearn.pkl"
EMISSION_SCALER_PATH = "Models_Trained/Emission_Scaler.pkl"

CARBON_PRICE_MODEL_PATH = "Models_Trained/CarbonPrice_Sklearn.pkl"
CARBON_PRICE_SCALER_PATH = "Models_Trained/CarbonPrice_Scaler.pkl"

RF_BUYSELL_PATH = "Models_Trained/RF_BuySell_Predictor.pkl"
RF_COMPLIANCE_PATH = "Models_Trained/RF_Compliance_Cost.pkl"
RF_COST_SAVINGS_PATH = "Models_Trained/RF_Cost_Savings.pkl"
RF_OPTIMIZATION_PATH = "Models_Trained/RF_Optimization_Scenario.pkl"


def build_fuel_one_hot(fuel_type):
    return (
        1.0 if fuel_type == "Mixed Fuel" else 0.0,
        1.0 if fuel_type == "Natural Gas" else 0.0,
        1.0 if fuel_type == "Renewable" else 0.0,
    )


def build_industry_one_hot(industry_type):
    return (
        1.0 if industry_type == "Energy" else 0.0,
        1.0 if industry_type == "Manufacturing" else 0.0,
        1.0 if industry_type == "Steel" else 0.0,
    )


def build_optimization_one_hot(scenario):
    return (
        1.0 if scenario == "Low Demand" else 0.0,
        1.0 if scenario == "Price Surge" else 0.0,
    )


def month_cyclical(month):
    angle = 2 * np.pi * (month / 12)
    return np.sin(angle), np.cos(angle)


def train_emission_model():
    df = pd.read_csv(DATA_PATH)

    X_cols = [
        "Energy_Demand_MWh",
        "Emission_Allowance_tCO2",
        "Carbon_Price_USD_per_t",
        "Emission_Intensity",
        "Fuel_Type_Mixed Fuel",
        "Fuel_Type_Natural Gas",
        "Fuel_Type_Renewable",
        "Industry_Type_Energy",
        "Industry_Type_Manufacturing",
        "Industry_Type_Steel",
    ]

    X = df[X_cols].astype(float).values
    y = df["Emission_Produced_tCO2"].values

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, EMISSION_MODEL_PATH)
    joblib.dump(scaler, EMISSION_SCALER_PATH)

    return model, scaler


def train_carbon_price_model():
    df = pd.read_csv(DATA_PATH)

    X_cols = [
        "Energy_Demand_MWh",
        "Emission_Produced_tCO2",
        "Emission_Allowance_tCO2",
        "Fuel_Type_Mixed Fuel",
        "Fuel_Type_Natural Gas",
        "Fuel_Type_Renewable",
        "Industry_Type_Energy",
        "Industry_Type_Manufacturing",
        "Industry_Type_Steel",
    ]

    X = df[X_cols].astype(float).values
    y = df["Carbon_Price_USD_per_t"].values

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, CARBON_PRICE_MODEL_PATH)
    joblib.dump(scaler, CARBON_PRICE_SCALER_PATH)

    return model, scaler


def load_model_and_scaler(model_path, scaler_path, train_fn):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return train_fn()


def load_model_if_exists(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def emission_category(emission):
    if emission <= 500:
        return "Low Emission"
    if emission <= 1000:
        return "Moderate Emission"
    if emission <= 2000:
        return "High Emission"
    return "Very High Emission"


def apply_custom_css():
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        [data-testid="stMain"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
        }
        
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
            font-weight: 400;
        }
        
        .metric-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        h1 {
            color: white !important;
            text-align: center;
            font-weight: 700 !important;
            font-size: 3rem !important;
            margin-bottom: 1rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        h2, h3 {
            color: #667eea !important;
            font-weight: 600 !important;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1.1rem;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }
        
        .input-section {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        
        .stSelectbox, .stNumberInput {
            margin-bottom: 1rem;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #667eea;
        }
        
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            margin: 1rem 0;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: white;
            border-radius: 10px 10px 0 0;
            padding: 10px 20px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)


def create_gauge_chart(value, title, max_value=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 20, 'color': '#667eea'}},
        delta={'reference': max_value * 0.7},
        gauge={
            'axis': {'range': [None, max_value], 'tickcolor': '#667eea'},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, max_value * 0.33], 'color': "#e8f5e9"},
                {'range': [max_value * 0.33, max_value * 0.66], 'color': "#fff9c4"},
                {'range': [max_value * 0.66, max_value], 'color': "#ffebee"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_pie_chart(values, labels, title):
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#667eea')),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def create_bar_chart(x, y, title, x_label, y_label):
    fig = go.Figure(data=[
        go.Bar(
            x=x,
            y=y,
            marker=dict(
                color=y,
                colorscale='Blues',
                showscale=True
            ),
            text=y,
            texttemplate='%{text:.2f}',
            textposition='outside'
        )
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#667eea')),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_heatmap(data, title):
    fig = px.imshow(
        data,
        labels=dict(color="Value"),
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#667eea')),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def create_comparison_chart(categories, current, benchmark, title):
    fig = go.Figure(data=[
        go.Bar(name='Current', x=categories, y=current, marker_color='#667eea'),
        go.Bar(name='Benchmark', x=categories, y=benchmark, marker_color='#764ba2')
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#667eea')),
        barmode='group',
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


st.set_page_config(
    page_title="Carbon Credit Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

st.markdown("<h1><i class='fas fa-leaf'></i> Carbon Credit Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem; margin-bottom: 2rem;'>AI-Powered Carbon Emission Analytics & Trading Intelligence</p>", unsafe_allow_html=True)

emission_model, emission_scaler = load_model_and_scaler(
    EMISSION_MODEL_PATH, EMISSION_SCALER_PATH, train_emission_model
)
carbon_price_model, carbon_price_scaler = load_model_and_scaler(
    CARBON_PRICE_MODEL_PATH, CARBON_PRICE_SCALER_PATH, train_carbon_price_model
)

rf_buysell = load_model_if_exists(RF_BUYSELL_PATH)
rf_compliance = load_model_if_exists(RF_COMPLIANCE_PATH)
rf_cost_savings = load_model_if_exists(RF_COST_SAVINGS_PATH)
rf_optimization = load_model_if_exists(RF_OPTIMIZATION_PATH)

with st.sidebar:
    st.markdown("### <i class='fas fa-sliders-h'></i> Configuration Panel", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("#### <i class='fas fa-chart-bar'></i> Energy Parameters", unsafe_allow_html=True)
    energy_demand = st.number_input("Energy Demand (MWh)", 0.0, 10000.0, 1000.0, key="energy")
    emission_allowance = st.number_input("Emission Allowance (tCO2)", 0.0, 10000.0, 500.0, key="allowance")
    carbon_price = st.number_input("Carbon Price (USD/t)", 0.0, 200.0, 50.0, key="cprice")
    emission_intensity = st.number_input("Emission Intensity", 0.0, 5.0, 0.5, key="intensity")
    month = st.number_input("Month (1-12)", 1, 12, 1, key="month")
    
    st.markdown("---")
    st.markdown("#### <i class='fas fa-cog'></i> Operational Parameters", unsafe_allow_html=True)
    fuel_type = st.selectbox("Fuel Type", ["Coal", "Mixed Fuel", "Natural Gas", "Renewable"], key="fuel")
    industry_type = st.selectbox("Industry Type", ["Construction", "Energy", "Manufacturing", "Steel"], key="industry")
    optimization_scenario = st.selectbox("Scenario", ["Normal", "Low Demand", "Price Surge"], key="scenario")
    
    st.markdown("---")
    predict_button = st.button("Run Prediction", type="primary")
    
    st.markdown("---")
    st.markdown("### <i class='fas fa-info-circle'></i> Model Info", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: #e8f5e9; padding: 1rem; border-radius: 10px; color: #2e7d32;'>
            <strong><i class='fas fa-check-circle'></i> Models Loaded:</strong><br>
            <i class='fas fa-check'></i> Emission Model<br>
            <i class='fas fa-check'></i> Carbon Price Model<br>
    """ + ("        <i class='fas fa-check'></i> Buy/Sell Model<br>\n" if rf_buysell else "") +
    ("        <i class='fas fa-check'></i> Compliance Model<br>\n" if rf_compliance else "") +
    ("        <i class='fas fa-check'></i> Cost Savings Model<br>\n" if rf_cost_savings else "") +
    ("        <i class='fas fa-check'></i> Optimization Model<br>\n" if rf_optimization else "") +
    """    </div>
    """, unsafe_allow_html=True)

if not predict_button:
    st.markdown("""
        <div class='card' style='text-align: center; padding: 4rem;'>
            <i class='fas fa-chart-line' style='font-size: 5rem; color: #667eea; margin-bottom: 2rem;'></i>
            <h2 style='color: #667eea;'>Welcome to Carbon Credit Analytics</h2>
            <p style='font-size: 1.1rem; color: #666; margin-top: 1rem;'>
                Configure your parameters in the sidebar and click <strong>Run Prediction</strong> to get comprehensive carbon credit insights with advanced visualizations.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-icon'><i class='fas fa-cloud'></i></div>
                <div class='metric-label'>Emission Prediction</div>
                <div style='font-size: 1rem; margin-top: 0.5rem;'>ML-based emission forecasting</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-icon'><i class='fas fa-chart-pie'></i></div>
                <div class='metric-label'>Visual Analytics</div>
                <div style='font-size: 1rem; margin-top: 0.5rem;'>Interactive charts & graphs</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-icon'><i class='fas fa-dollar-sign'></i></div>
                <div class='metric-label'>Cost Analysis</div>
                <div style='font-size: 1rem; margin-top: 0.5rem;'>Compliance & savings insights</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.stop()

with st.spinner('Running AI models...'):
    fuel_type_mixed, fuel_type_natural_gas, fuel_type_renewable = build_fuel_one_hot(fuel_type)
    industry_type_energy, industry_type_manufacturing, industry_type_steel = build_industry_one_hot(industry_type)
    optimization_low, optimization_surge = build_optimization_one_hot(optimization_scenario)

    emission_input = np.array([[
        energy_demand, emission_allowance, carbon_price, emission_intensity,
        fuel_type_mixed, fuel_type_natural_gas, fuel_type_renewable,
        industry_type_energy, industry_type_manufacturing, industry_type_steel
    ]])
    emission_scaled = emission_scaler.transform(emission_input)
    predicted_emission = emission_model.predict(emission_scaled)[0]

    carbon_price_input = np.array([[
        energy_demand, predicted_emission, emission_allowance,
        fuel_type_mixed, fuel_type_natural_gas, fuel_type_renewable,
        industry_type_energy, industry_type_manufacturing, industry_type_steel
    ]])
    carbon_price_scaled = carbon_price_scaler.transform(carbon_price_input)
    predicted_carbon_price = carbon_price_model.predict(carbon_price_scaled)[0]
    
    net_position = emission_allowance - predicted_emission
    transaction_action = "N/A"
    compliance_pred = 0
    cost_savings_pred = 0
    optimization_pred = 0
    
    if rf_buysell is not None:
        net_position_price_interaction = net_position * carbon_price
        buysell_input = np.array([[
            energy_demand, predicted_emission, emission_allowance, carbon_price,
            net_position, net_position_price_interaction,
            fuel_type_mixed, fuel_type_natural_gas, fuel_type_renewable
        ]])
        buysell_pred = rf_buysell.predict(buysell_input)[0]
        transaction_action = "Buy" if buysell_pred == 0 else "Sell"

    if rf_compliance is not None:
        compliance_input = np.array([[
            energy_demand, predicted_emission, carbon_price,
            industry_type_energy, industry_type_manufacturing, industry_type_steel
        ]])
        compliance_pred = rf_compliance.predict(compliance_input)[0]

    if rf_cost_savings is not None:
        cost_savings_input = np.array([[
            energy_demand, predicted_emission,
            optimization_low, optimization_surge,
            fuel_type_mixed, fuel_type_natural_gas, fuel_type_renewable
        ]])
        cost_savings_pred = rf_cost_savings.predict(cost_savings_input)[0]

    if rf_optimization is not None:
        month_sin, month_cos = month_cyclical(month)
        optimization_input = np.array([[
            energy_demand, month_sin, month_cos,
            fuel_type_mixed, fuel_type_natural_gas, fuel_type_renewable,
            industry_type_energy, industry_type_manufacturing, industry_type_steel
        ]])
        optimization_pred = rf_optimization.predict(optimization_input)[0]

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'><i class='fas fa-smog'></i></div>
            <div class='metric-label'>Carbon Emission</div>
            <div class='metric-value'>{predicted_emission:.1f}</div>
            <div style='font-size: 0.8rem;'>tCO2</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
            <div class='metric-icon'><i class='fas fa-dollar-sign'></i></div>
            <div class='metric-label'>Carbon Price</div>
            <div class='metric-value'>${np.absolute(predicted_carbon_price):.1f}</div>
            <div style='font-size: 0.8rem;'>USD/t</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    icon = "arrow-down" if transaction_action == "Buy" else "arrow-up" if transaction_action == "Sell" else "minus"
    color = "#4facfe" if transaction_action == "Buy" else "#f5576c" if transaction_action == "Sell" else "#ffd700"
    st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, {color} 0%, {color}dd 100%);'>
            <div class='metric-icon'><i class='fas fa-{icon}'></i></div>
            <div class='metric-label'>Action</div>
            <div class='metric-value' style='font-size: 2rem;'>{transaction_action}</div>
            <div style='font-size: 0.8rem;'>Credits</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    category = emission_category(predicted_emission)
    category_colors = {
        "Low Emission": "#4caf50",
        "Moderate Emission": "#ff9800",
        "High Emission": "#ff5722",
        "Very High Emission": "#d32f2f"
    }
    st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, {category_colors.get(category, "#667eea")} 0%, {category_colors.get(category, "#764ba2")}dd 100%);'>
            <div class='metric-icon'><i class='fas fa-thermometer-half'></i></div>
            <div class='metric-label'>Category</div>
            <div class='metric-value' style='font-size: 1.3rem;'>{category.replace(" Emission", "")}</div>
            <div style='font-size: 0.8rem;'>Emission Level</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Overview", "Analytics", "Financial"])

with tab1:
    st.markdown("### <i class='fas fa-chart-line'></i> Emission & Price Analysis", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gauge_fig = create_gauge_chart(predicted_emission, "Emission Level (tCO2)", max_value=2500)
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        pie_data = [predicted_emission, emission_allowance, max(0, emission_allowance - predicted_emission)]
        pie_labels = ['Produced', 'Allowance', 'Surplus']
        pie_fig = create_pie_chart(pie_data, pie_labels, "Emission Distribution")
        st.plotly_chart(pie_fig, use_container_width=True)

with tab2:
    st.markdown("### <i class='fas fa-chart-bar'></i> Detailed Analytics", unsafe_allow_html=True)
    
    categories = ['Emission\nProduced', 'Emission\nAllowance', 'Carbon\nPrice x10']
    values = [predicted_emission, emission_allowance, np.absolute(predicted_carbon_price) * 10]
    bar_fig = create_bar_chart(categories, values, "Operational Metrics", "Metrics", "Value")
    st.plotly_chart(bar_fig, use_container_width=True)

with tab3:
    st.markdown("### <i class='fas fa-money-bill-wave'></i> Financial Analysis", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
                <div class='metric-icon'><i class='fas fa-receipt'></i></div>
                <div class='metric-label'>Compliance Cost</div>
                <div class='metric-value'>${compliance_pred:.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);'>
                <div class='metric-icon'><i class='fas fa-piggy-bank'></i></div>
                <div class='metric-label'>Cost Savings</div>
                <div class='metric-value'>${cost_savings_pred:.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        net_financial = cost_savings_pred - compliance_pred
        net_color = "#38ef7d" if net_financial > 0 else "#ff6a00"
        st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, {net_color} 0%, {net_color}dd 100%);'>
                <div class='metric-icon'><i class='fas fa-balance-scale'></i></div>
                <div class='metric-label'>Net Financial</div>
                <div class='metric-value'>${net_financial:.0f}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("""
    <div class='card' style='text-align: center; padding: 1.5rem; margin-top: 2rem;'>
        <p style='color: #667eea; font-weight: 600; margin: 0;'>
            <i class='fas fa-info-circle'></i> AI-Powered Predictions | 
            <i class='fas fa-database'></i> Data-Driven Insights | 
            <i class='fas fa-shield-alt'></i> Secure & Reliable
        </p>
    </div>
""", unsafe_allow_html=True)

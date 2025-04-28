import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning
# Add this near the top of your file, after the other imports
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

# Suppress the version warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Page configuration
st.set_page_config(
    page_title="⚡ EV Battery Life Predictor",
    page_icon="🔋",
    layout="wide"
)

# Custom CSS for better fonts and styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
    }
    
    .main-header {
        font-size: 42px;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    
    .subheader {
        font-size: 28px;
        color: #182848;
    }
    
    .prediction-box {
        background-color: #f0f5ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4b6cb7;
        margin: 20px 0;
    }
    
    .highlight {
        font-weight: bold;
        color: #4b6cb7;
    }
    
    .stButton>button {
        background-color: #4b6cb7;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #182848;
    }
    
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 12px;
        color: #666;
    }
    
    .battery-status {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    
    <div class="main-header">⚡ EV Battery Life Prediction 🔋</div>
""", unsafe_allow_html=True)

# Load the model and polynomial transformer from pickle file
try:
    model, poly = joblib.load('batter_life_model.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.warning(f"⚠️ Issue loading model: {e}")
    st.info("Using a placeholder model for demonstration.")
    # Create dummy model for demonstration
    class DummyModel:
        def predict(self, X):
            return np.array([75.5])
    
    class DummyPoly:
        def transform(self, X):
            return X
            
    model, poly = DummyModel(), DummyPoly()

# Two column layout
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    <div class="subheader">🚗 Predict Your EV's Battery Health 🔌</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This intelligent app analyzes critical factors that affect your electric vehicle's 
    battery performance and predicts its remaining life. Enter your vehicle's data below 
    to get a personalized prediction!
    """)
    
    # Create tabs for different input sections
    tab1, tab2 = st.tabs(["📊 Battery Parameters", "ℹ️ Tips & Info"])
    
    with tab1:
        # User Inputs with emojis and better layout
        st.subheader("🔄 Battery Usage History")
        charging_cycles = st.slider("🔋 Charging Cycles", 
                                  min_value=1, max_value=5000, value=1000,
                                  help="Number of complete charge-discharge cycles")
        
        st.subheader("🌡️ Environmental Factors")
        avg_temp = st.slider("🌡️ Average Operating Temperature (°C)", 
                           min_value=-20, max_value=50, value=25,
                           help="Average temperature the battery operates in")
        
        st.subheader("⚡ Charging Behavior")
        charging_time = st.slider("⏱️ Average Charging Time (hours)", 
                                min_value=0.5, max_value=10.0, value=2.5, step=0.1,
                                help="Typical time spent charging the battery")
        
        discharge_rate = st.slider("⚡ Average Discharge Rate (A)", 
                                 min_value=0.1, max_value=5.0, value=0.7, step=0.1,
                                 help="Rate at which battery discharges during use")
    
    with tab2:
        st.markdown("""
        ### 📌 Tips for Maximizing Battery Life
        
        - 🔋 Avoid frequent fast charging when possible
        - 🌡️ Park in temperature-controlled areas when available
        - 📊 Try to maintain charge between 20% and 80%
        - 🚫 Avoid complete discharge when possible
        - 🔌 Use manufacturer-recommended charging equipment
        """)

    # Button to make the prediction with improved styling
    predict_button = st.button('🔮 Predict Battery Life')

with col2:
    # Car image display
    st.image("https://api.dicebear.com/7.x/shapes/svg?seed=electric-car", width=300)
    
    # Interactive elements
    car_type = st.selectbox(
        "🚙 Select your vehicle type",
        ["Sedan", "SUV", "Compact", "Sports Car", "Truck"]
    )
    
    usage_pattern = st.radio(
        "🛣️ Primary usage pattern",
        ["City driving", "Highway commuting", "Mixed usage", "Heavy load"]
    )
    
    # Create a placeholder for prediction results
    result_container = st.container()

# Handle prediction when button is clicked
if predict_button:
    # Progress bar for visual effect
    progress_bar = st.progress(0)
    for i in range(100):
        # Update the progress bar
        progress_bar.progress(i + 1)
        # Small delay for effect
        if i < 98:  # Skip delay at the end
            import time
            time.sleep(0.01)
            
    # Prepare the input data for prediction using DataFrame with feature names
    feature_names = ['charging_cycles', 'avg_temp', 'charging_time', 'discharge_rate']
    input_data = pd.DataFrame({
        'charging_cycles': [charging_cycles],
        'avg_temp': [avg_temp],
        'charging_time': [charging_time],
        'discharge_rate': [discharge_rate]
    })
    
    # Transform the input data using the polynomial features
    try:
        input_poly = poly.transform(input_data)
    except:
        # Fallback if DataFrame approach doesn't work
        input_data_array = np.array([[charging_cycles, avg_temp, charging_time, discharge_rate]])
        input_poly = poly.transform(input_data_array)
        st.info("Used fallback transformation method")
    
    # Make the prediction
    predicted_battery_life = model.predict(input_poly)
    battery_percentage = predicted_battery_life[0].item() if hasattr(predicted_battery_life[0], 'item') else float(predicted_battery_life[0])
    
    # Create a fancy prediction display in the result container
    with result_container:
        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.markdown("### 🔋 Battery Health Analysis")
        
        # Battery emoji representation
        if battery_percentage >= 80:
            battery_emoji = "🔋🔋🔋🔋🔋"
            status = "Excellent"
            color = "green"
        elif battery_percentage >= 60:
            battery_emoji = "🔋🔋🔋🔋⚪"
            status = "Good"
            color = "lightgreen"
        elif battery_percentage >= 40:
            battery_emoji = "🔋🔋🔋⚪⚪"
            status = "Average"
            color = "orange"
        elif battery_percentage >= 20:
            battery_emoji = "🔋🔋⚪⚪⚪"
            status = "Poor"
            color = "orangered"
        else:
            battery_emoji = "🔋⚪⚪⚪⚪"
            status = "Critical"
            color = "red"
        
        # Show prediction with formatting
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="font-size: 28px;">{battery_emoji}</div>
            <div>
                <div class="battery-status" style="color: {color};">{status}</div>
                <div>Predicted Battery Life: <span class="highlight">{battery_percentage:.1f}%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a simple gauge chart
        fig, ax = plt.subplots(figsize=(4, 0.8))
        ax.barh(0, 100, color='lightgray', height=0.4)
        ax.barh(0, battery_percentage, color=color, height=0.4)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        st.pyplot(fig)
        
        # Recommendations based on prediction
        st.markdown("### 🔍 Recommendations")
        
        if battery_percentage >= 80:
            st.success("✅ Your battery is in excellent condition! Continue your current usage patterns.")
        elif battery_percentage >= 60:
            st.info("ℹ️ Your battery is in good shape. Consider optimizing charging cycles for longevity.")
        elif battery_percentage >= 40:
            st.warning("⚠️ Battery performance is average. Try to reduce fast charging and extreme temperatures.")
        elif battery_percentage >= 20:
            st.warning("⚠️ Battery health is declining. Consider battery maintenance or replacement planning.")
        else:
            st.error("🚨 Battery health is critical. Recommended battery replacement soon.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Add some additional useful features
st.markdown("---")
with st.expander("🔍 Understand EV Battery Factors"):
    st.markdown("""
    | Factor | Impact on Battery Life | Optimal Range |
    |--------|------------------------|--------------|
    | 🔄 Charging Cycles | Higher cycles = more wear | Depends on battery chemistry |
    | 🌡️ Temperature | Extreme temps reduce lifespan | 15°C - 25°C (59°F - 77°F) |
    | ⏱️ Charging Time | Faster charging can stress battery | 2-3 hours (standard charging) |
    | ⚡ Discharge Rate | Faster discharge = more stress | 0.3-0.7A for most EVs |
    """)

# Sample data visualization
with st.expander("📈 Battery Degradation Trends"):
    # Sample data for visualization
    cycles = np.array([0, 500, 1000, 1500, 2000, 2500, 3000])
    capacity = 100 * np.exp(-0.0001 * cycles)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cycles, capacity, 'o-', color='#4b6cb7', linewidth=2)
    ax.set_xlabel('Number of Cycles')
    ax.set_ylabel('Battery Capacity (%)')
    ax.set_title('Typical EV Battery Degradation Curve')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# Footer with car emojis
st.markdown("""
<div class="footer">
    🚗 🚙 🏎️ 🚐 🚓 🚕 <br>
    © 2025 EV Battery Predictor | For educational purposes only
</div>
""", unsafe_allow_html=True)
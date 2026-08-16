import streamlit as st
import pandas as pd
import joblib
import time

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Smart Traffic AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("accident_severity_model.pkl")
    model_features = joblib.load("model_features.pkl")
    return model, model_features

try:
    model, model_features = load_model()
except Exception as e:
    st.error("Model files could not be loaded.")
    st.code(str(e))
    st.stop()

# ==========================================
# SAFE PROFESSIONAL STYLING
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #f6f8fc;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}

div.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 600;
}

.metric-card {
    background: white;
    border-radius: 16px;
    padding: 18px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.title("🚦 Smart Traffic Accident Severity Prediction")

st.caption(
    "AI-powered machine learning system that predicts accident severity "
    "using driver, vehicle, road, weather and traffic conditions."
)

st.divider()

# ==========================================
# PROJECT STATS
# ==========================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Dataset Records", "12,316")

with c2:
    st.metric("Model Features", "112")

with c3:
    st.metric("Severity Classes", "3")

with c4:
    st.metric("Best Accuracy", "85.1%")

st.divider()

# ==========================================
# DRIVER & VEHICLE
# ==========================================

st.subheader("👨‍🚗 Driver & Vehicle Information")

col1, col2 = st.columns(2)

with col1:

    hour = st.slider("Hour of Day", 0, 23, 12)

    day = st.selectbox(
        "Day of Week",
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    )

    age = st.selectbox(
        "Age Band of Driver",
        ["18-30","31-50","Over 51","Under 18","Unknown"]
    )

    sex = st.selectbox(
        "Sex of Driver",
        ["Male","Female","Unknown"]
    )

    experience = st.selectbox(
        "Driving Experience",
        ["Below 1yr","1-2yr","2-5yr","5-10yr","Above 10yr","No Licence","Unknown"]
    )

    vehicle = st.selectbox(
        "Type of Vehicle",
        [
            "Automobile",
            "Lorry (41?100Q)",
            "Lorry (11?40Q)",
            "Public (> 45 seats)",
            "Public (13?45 seats)",
            "Public (12 seats)",
            "Taxi",
            "Motorcycle",
            "Other"
        ]
    )

    area = st.selectbox(
        "Accident Area",
        ["Residential areas","Office areas","Industrial areas","Other"]
    )

    road_surface = st.selectbox(
        "Road Surface Type",
        ["Asphalt roads","Earth roads","Gravel roads","Other"]
    )

with col2:

    road_condition = st.selectbox(
        "Road Surface Condition",
        ["Dry","Wet or damp","Snow","Flood over 3cm. deep"]
    )

    light = st.selectbox(
        "Light Condition",
        [
            "Daylight",
            "Darkness - lights lit",
            "Darkness - lights unlit",
            "Darkness - no lighting"
        ]
    )

    weather = st.selectbox(
        "Weather Condition",
        [
            "Normal",
            "Raining",
            "Raining and Windy",
            "Cloudy",
            "Windy",
            "Fog or mist",
            "Snow",
            "Other",
            "Unknown"
        ]
    )

    collision = st.selectbox(
        "Type of Collision",
        [
            "Vehicle with vehicle collision",
            "Collision with roadside objects",
            "Collision with pedestrians",
            "Collision with animals",
            "Rollover",
            "Other"
        ]
    )

    vehicles = st.number_input(
        "Number of Vehicles Involved",
        1, 20, 2
    )

    casualties = st.number_input(
        "Number of Casualties",
        0, 20, 1
    )

    movement = st.selectbox(
        "Vehicle Movement",
        [
            "Going straight",
            "Moving Backward",
            "Overtaking",
            "Changing lane to the left",
            "Changing lane to the right",
            "Turnover",
            "Other"
        ]
    )

    cause = st.selectbox(
        "Cause of Accident",
        [
            "No distancing",
            "Changing lane to the left",
            "Changing lane to the right",
            "Driving carelessly",
            "Driving under the influence of drugs",
            "Driving to the left",
            "Driving at high speed",
            "Overtaking",
            "Other"
        ]
    )

st.divider()

# ==========================================
# CREATE INPUT
# ==========================================

input_data = pd.DataFrame({
    "Hour": [hour],
    "Day_of_week": [day],
    "Age_band_of_driver": [age],
    "Sex_of_driver": [sex],
    "Driving_experience": [experience],
    "Type_of_vehicle": [vehicle],
    "Area_accident_occured": [area],
    "Road_surface_type": [road_surface],
    "Road_surface_conditions": [road_condition],
    "Light_conditions": [light],
    "Weather_conditions": [weather],
    "Type_of_collision": [collision],
    "Number_of_vehicles_involved": [vehicles],
    "Number_of_casualties": [casualties],
    "Vehicle_movement": [movement],
    "Cause_of_accident": [cause]
})

# ==========================================
# ENCODE
# ==========================================

input_encoded = pd.get_dummies(input_data, dtype=int)

input_encoded = input_encoded.reindex(
    columns=model_features,
    fill_value=0
)

# ==========================================
# PREDICTION
# ==========================================

st.subheader("🎯 Prediction")

if st.button("🚦 Analyze Accident Severity", type="primary"):

    with st.spinner("AI model is analyzing the conditions..."):
        time.sleep(1)
        prediction = model.predict(input_encoded)[0]

    st.divider()

    if prediction == "Fatal injury":

        st.error("🔴 **Predicted Severity: Fatal Injury**")

        st.warning(
            "The model indicates a potentially very severe accident outcome."
        )

    elif prediction == "Serious Injury":

        st.warning("🟠 **Predicted Severity: Serious Injury**")

        st.info(
            "The model indicates a potentially serious accident outcome."
        )

    else:

        st.success("🟢 **Predicted Severity: Slight Injury**")

        st.info(
            "The model predicts a comparatively lower accident severity."
        )

    st.caption(
        "Prediction generated using the trained machine learning model."
    )

st.divider()

st.caption(
    "Smart Traffic Accident Severity Prediction | "
    "Python • Pandas • Scikit-learn • Random Forest • Streamlit"
)
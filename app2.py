import streamlit as st
import pandas as pd
import joblib  # For loading the trained model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("ideal_predict_model_1.joblib")  # Replace with the actual path to your model

model = load_model()

# Pre-trained encoders for categorical features
@st.cache_resource
def load_encoders():
    encoders = {
        "ProductSize": LabelEncoder(),
        "fiSecondaryDesc": LabelEncoder(),
        "Enclosure": LabelEncoder(),
        "fiProductClassDesc": LabelEncoder(),
    }
    # Fit the encoders with the same data used during training
    encoders["ProductSize"].fit(["Mini", "Small", "Compact", "Medium", "Large / Medium", "Large"])
    encoders["fiSecondaryDesc"].fit(["C", "B", "G", "H", "E", "D", "F", "K", "L", "A", "M", "J", "P"])  # Add all categories as needed
    encoders["Enclosure"].fit([
        "OROPS", "EROPS", "EROPS w AC", "EROPS AC", "NO ROPS", "None or Unspecified"
    ])
    encoders["fiProductClassDesc"].fit([
        "Backhoe Loader - 14.0 to 15.0 Ft Standard Digging Depth",
        "Track Type Tractor, Dozer - 20.0 to 75.0 Horsepower",
        "Wheel Loader - 150.0 to 175.0 Horsepower",
        "Track Type Tractor, Dozer - 85.0 to 105.0 Horsepower",
        "Hydraulic Excavator, Track - 21.0 to 24.0 Metric Tons",
        "Track Type Tractor, Dozer - 130.0 to 160.0 Horsepower",
        "Hydraulic Excavator, Track - 12.0 to 14.0 Metric Tons",
        "Track Type Tractor, Dozer - 260.0 + Horsepower",
        "Wheel Loader - 120.0 to 135.0 Horsepower",
        "Backhoe Loader - 15.0 to 16.0 Ft Standard Digging Depth",
        "Skid Steer Loader - 1351.0 to 1601.0 Lb Operating Capacity",
        "Skid Steer Loader - 1601.0 to 1751.0 Lb Operating Capacity",
        "Motorgrader - 145.0 to 170.0 Horsepower",
        "Hydraulic Excavator, Track - 33.0 to 40.0 Metric Tons",
        "Skid Steer Loader - 1751.0 to 2201.0 Lb Operating Capacity",
        "Track Type Tractor, Dozer - 160.0 to 190.0 Horsepower",
        "Wheel Loader - 175.0 to 200.0 Horsepower",
        "Skid Steer Loader - 1251.0 to 1351.0 Lb Operating Capacity",
        "Hydraulic Excavator, Track - 6.0 to 8.0 Metric Tons",
        "Motorgrader - 45.0 to 130.0 Horsepower",
        "Wheel Loader - 100.0 to 110.0 Horsepower",
        "Track Type Tractor, Dozer - 75.0 to 85.0 Horsepower",
        "Hydraulic Excavator, Track - 40.0 to 50.0 Metric Tons",
        "Hydraulic Excavator, Track - 19.0 to 21.0 Metric Tons"
    ])  # Add all categories as needed
    return encoders

encoders = load_encoders()

# Streamlit app
st.title("Bulldozer Price Prediction")
st.write("Enter the details of the bulldozer to predict its price:")

# Input fields for the top 6 features
YearMade = st.number_input("Year Made", min_value=1900, max_value=2025, step=1, value=2000)

ProductSize = st.selectbox(
    "Product Size",
    ["Mini", "Small", "Compact", "Medium", "Large / Medium", "Large"]
)

saleYear = st.number_input("Sale Year", min_value=1980, max_value=2050, step=1, value=2024)

fiSecondaryDesc = st.selectbox(
    "Secondary Description (e.g., Bulldozer Type)", 
    ["C", "B", "G", "H", "E", "D", "F", "K", "L", "A", "M", "J", "P"]
)

Enclosure = st.selectbox(
    "Enclosure (Machine Configuration)",
    ["OROPS", "EROPS", "EROPS w AC", "EROPS AC", "NO ROPS", "None or Unspecified"]
)

fiProductClassDesc = st.selectbox(
    "Product Class Description (e.g., Track Type)", 
    [
        "Backhoe Loader - 14.0 to 15.0 Ft Standard Digging Depth",
        "Track Type Tractor, Dozer - 20.0 to 75.0 Horsepower",
        "Wheel Loader - 150.0 to 175.0 Horsepower",
        "Track Type Tractor, Dozer - 85.0 to 105.0 Horsepower",
        "Hydraulic Excavator, Track - 21.0 to 24.0 Metric Tons",
        "Track Type Tractor, Dozer - 130.0 to 160.0 Horsepower",
        "Hydraulic Excavator, Track - 12.0 to 14.0 Metric Tons",
        "Track Type Tractor, Dozer - 260.0 + Horsepower",
        "Wheel Loader - 120.0 to 135.0 Horsepower",
        "Backhoe Loader - 15.0 to 16.0 Ft Standard Digging Depth",
        "Skid Steer Loader - 1351.0 to 1601.0 Lb Operating Capacity",
        "Skid Steer Loader - 1601.0 to 1751.0 Lb Operating Capacity",
        "Motorgrader - 145.0 to 170.0 Horsepower",
        "Hydraulic Excavator, Track - 33.0 to 40.0 Metric Tons",
        "Skid Steer Loader - 1751.0 to 2201.0 Lb Operating Capacity",
        "Track Type Tractor, Dozer - 160.0 to 190.0 Horsepower",
        "Wheel Loader - 175.0 to 200.0 Horsepower",
        "Skid Steer Loader - 1251.0 to 1351.0 Lb Operating Capacity",
        "Hydraulic Excavator, Track - 6.0 to 8.0 Metric Tons",
        "Motorgrader - 45.0 to 130.0 Horsepower",
        "Wheel Loader - 100.0 to 110.0 Horsepower",
        "Track Type Tractor, Dozer - 75.0 to 85.0 Horsepower",
        "Hydraulic Excavator, Track - 40.0 to 50.0 Metric Tons",
        "Hydraulic Excavator, Track - 19.0 to 21.0 Metric Tons"
    ]
)

# Predict button
if st.button("Predict Price"):
    # Create a dataframe with the input values
    input_data = pd.DataFrame({
        "YearMade": [YearMade],
        "ProductSize": [encoders["ProductSize"].transform([ProductSize])[0]],
        "saleYear": [saleYear],
        "fiSecondaryDesc": [encoders["fiSecondaryDesc"].transform([fiSecondaryDesc])[0]],
        "Enclosure": [encoders["Enclosure"].transform([Enclosure])[0]],
        "fiProductClassDesc": [encoders["fiProductClassDesc"].transform([fiProductClassDesc])[0]]
    })

    # Ensure feature alignment (if necessary)
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make a prediction
    predicted_price = model.predict(input_data)
    
    # Display the prediction
    st.subheader(f"Predicted Price: ${predicted_price[0]:,.2f}",divider='green')
    #st.markdown(f'<h3 style="color:white; display:inline;">Predicted Price : </h3><h2 style="color:green; font-size:40px; display:inline;">${predicted_price[0]:,.2f}</h2>', unsafe_allow_html=True)
    




import streamlit as st
import pandas as pd
import joblib

import sklearn
import streamlit as st

st.write("Scikit-Learn Version:", sklearn.__version__)

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="darpan1703/gl_tourism_project_app",
    filename="best_model.pkl"
)

model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction")

age = st.number_input("Age", 18, 100, 35)

monthly_income = st.number_input(
    "Monthly Income",
    1000,
    100000,
    25000
)

city_tier = st.selectbox(
    "City Tier",
    [1,2,3]
)

if st.button("Predict"):

    input_df = pd.DataFrame({

        "Age":[age],
        "TypeofContact":["Self Enquiry"],
        "CityTier":[city_tier],
        "Occupation":["Salaried"],
        "Gender":["Male"],
        "NumberOfPersonVisiting":[2],
        "PreferredPropertyStar":[3],
        "MaritalStatus":["Single"],
        "NumberOfTrips":[3],
        "Passport":[1],
        "OwnCar":[1],
        "NumberOfChildrenVisiting":[0],
        "Designation":["Executive"],
        "MonthlyIncome":[monthly_income],
        "PitchSatisfactionScore":[4],
        "ProductPitched":["Deluxe"],
        "NumberOfFollowups":[2],
        "DurationOfPitch":[15]

    })

    prediction = model.predict(input_df)

    st.success(
        f"Prediction: {prediction[0]}"
    )

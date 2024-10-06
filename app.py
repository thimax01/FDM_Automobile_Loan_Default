import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('loan_model.pkl')

# Set up the Streamlit app with a sidebar and title
st.set_page_config(page_title="Loan Default Prediction", page_icon="💳", layout='centered')
st.title("💳 Loan Default Prediction App")

# Add a sidebar for information and instructions
st.sidebar.title("App Instructions")
st.sidebar.write(
    """
    This app predicts whether a client will default on their loan based on various input features. 
    Adjust the sliders and choose options from dropdowns and radio buttons to enter client details.
    """
)

# Create sliders and other inputs for user input
Client_Income = st.slider("Client Income", min_value=0, max_value=1000000, step=1000, value=0)
Child_Count = st.slider("Child Count", min_value=0, max_value=10, step=1, value=0)
Credit_Amount = st.slider("Credit Amount", min_value=0, max_value=10000000, step=10000, value=0)
Loan_Annuity = st.slider("Loan Annuity", min_value=0, max_value=50000, step=500, value=0)
Registration_years = st.slider("Registration Years", min_value=0, max_value=100, step=1, value=5)

# User inputs for categorical variables (using radio buttons for binary options and dropdowns for multiple choice)
Client_Education = st.selectbox("Client Education", ['Graduation', 'Graduation dropout', 'Junior secondary', 'Post Grad', 'Secondary'])
Client_Income_Type = st.selectbox("Client Income Type", ['Commercial', 'Govt Job', 'Maternity leave', 'Retired', 'Service', 'Student'])
Client_Marital_Status = st.selectbox("Client Marital Status", ['D', 'M', 'S', 'W'])
Client_Gender = st.radio("Client Gender", ['Female', 'Male'])  # Radio button for binary choice
Loan_Contract_Type = st.radio("Loan Contract Type", ['CL', 'RL'])  # Radio button for binary choice
Car_Owned = st.radio("Car Owned", ['No', 'Yes'])  # Radio button for binary choice
Bike_Owned = st.radio("Bike Owned", ['No', 'Yes'])  # Radio button for binary choice
Active_Loan = st.radio("Active Loan", ['No', 'Yes'])  # Radio button for binary choice
House_Own = st.radio("House Owned", ['No', 'Yes'])  # Radio button for binary choice
Client_Permanent_Match_Tag = st.radio("Client Permanent Match Tag", ['No', 'Yes'])  # Radio button for binary choice
Client_Contact_Work_Tag = st.radio("Client Contact Work Tag", ['No', 'Yes'])  # Radio button for binary choice

# Additional input fields using sliders
Client_Family_Members = st.slider("Client Family Members", min_value=0, max_value=10, step=1, value=1)
Age_Years = st.slider("Age (Years)", min_value=0, max_value=100, step=1, value=30)
Employed_Years = st.slider("Employed (Years)", min_value=0, max_value=50, step=1, value=5)
ID_years = st.slider("ID Years", min_value=0, max_value=50, step=1, value=5)

# Define functions to encode categorical features
def encode_client_education(education):
    education_mapping = {
        'Graduation': 0,
        'Graduation dropout': 1,
        'Junior secondary': 2,
        'Post Grad': 3,
        'Secondary': 4
    }
    return education_mapping.get(education, -1)

def encode_Client_Income_Type(income):
    income_mapping = {
        'Commercial': 0,
        'Govt Job': 1,
        'Maternity leave': 2,
        'Retired': 3,
        'Service': 4,
        'Student': 5
    }
    return income_mapping.get(income, -1)

def encode_Client_Marital_Status(marital_status):
    marital_status_mapping = {
        'D': 0,
        'M': 1,
        'S': 2,
        'W': 3
    }
    return marital_status_mapping.get(marital_status, -1)

def encode_Client_Gender(gender):
    gender_mapping = {
        'Female': 0,
        'Male': 1
    }
    return gender_mapping.get(gender, -1)

def encode_Loan_Contract_Type(contract_type):
    contract_type_mapping = {
        'CL': 0,
        'RL': 1
    }
    return contract_type_mapping.get(contract_type, -1)

def encode_Car_Owned(Car_Owned):
    Car_Owned_mapping = {
        'No': 0,
        'Yes': 1
    }
    return Car_Owned_mapping.get(Car_Owned, -1)

def encode_Bike_Owned(Bike_Owned):
    Bike_Owned_mapping = {
        'No': 0,
        'Yes': 1
    }
    return Bike_Owned_mapping.get(Bike_Owned, -1)

def encode_Active_Loan(Active_Loan):
    Active_Loan_mapping = {
        'No': 0,
        'Yes': 1
    }
    return Active_Loan_mapping.get(Active_Loan, -1)

def encode_House_Own(House_Own):
    House_Own_mapping = {
        'No': 0,
        'Yes': 1
    }
    return House_Own_mapping.get(House_Own, -1)

def encode_Client_Permanent_Match_Tag(tag):
    tag_mapping = {
        'No': 0,
        'Yes': 1
    }
    return tag_mapping.get(tag, -1)

def encode_Client_Contact_Work_Tag(tag):
    tag_mapping = {
        'No': 0,
        'Yes': 1
    }
    return tag_mapping.get(tag, -1)

# Function to encode all input data
def encode_input():
    education_encoded = encode_client_education(Client_Education)
    income_encoded = encode_Client_Income_Type(Client_Income_Type)
    marital_status_encoded = encode_Client_Marital_Status(Client_Marital_Status)
    gender_encoded = encode_Client_Gender(Client_Gender)
    contract_type_encoded = encode_Loan_Contract_Type(Loan_Contract_Type)
    car_owned_encoded = encode_Car_Owned(Car_Owned)
    bike_owned_encoded = encode_Bike_Owned(Bike_Owned)
    active_loan_encoded = encode_Active_Loan(Active_Loan)
    house_owned_encoded = encode_House_Own(House_Own)
    permanent_match_tag_encoded = encode_Client_Permanent_Match_Tag(Client_Permanent_Match_Tag)
    contact_work_tag_encoded = encode_Client_Contact_Work_Tag(Client_Contact_Work_Tag)

    # Combine all features into one array
    input_features = [
        Client_Income, Child_Count, Credit_Amount, Loan_Annuity, Registration_years, Client_Family_Members,
        permanent_match_tag_encoded, contact_work_tag_encoded, Age_Years,
        Employed_Years, ID_years, education_encoded, income_encoded,
        marital_status_encoded, gender_encoded, contract_type_encoded,
        car_owned_encoded, bike_owned_encoded, active_loan_encoded, house_owned_encoded
    ]

    return np.array(input_features).reshape(1, -1)

# Make prediction when button is clicked
if st.button("Predict"):
    # Preprocess input
    input_data = encode_input()

    # Get prediction probabilities
    prediction_proba = model.predict_proba(input_data)

    # Define your own threshold based on desired sensitivity/specificity
    threshold = 0.05  

    # Make prediction based on the threshold
    prediction = (prediction_proba[0][1] >= threshold).astype(int)

    # Display result
    if prediction == 0:
        st.success("The client is predicted to NOT default on the loan. 🎉")
    else:
        st.error("The client is predicted to default on the loan. ⚠️")

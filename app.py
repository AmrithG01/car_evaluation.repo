import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Car Evaluation ML", page_icon="🚗", layout="centered")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 48px;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 18px;
        margin-bottom: 40px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #4ECDC4;
        color: white;
        font-size: 20px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45b7af;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Car Evaluation AI 🚗</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter your car attributes below to predict its acceptability rating</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open("car_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("Model file `car_model.pkl` not found! Please run `python train_model.py` to train and save the model.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    buying = st.selectbox("Buying Price", ["vhigh", "high", "med", "low"], help="Purchase price of the car")
    doors = st.selectbox("Number of Doors", ["2", "3", "4", "5more"])
    lug_boot = st.selectbox("Luggage Boot Size", ["small", "med", "big"])

with col2:
    maint = st.selectbox("Maintenance Price", ["vhigh", "high", "med", "low"], help="Cost of maintenance")
    persons = st.selectbox("Capacity (Persons)", ["2", "4", "more"])
    safety = st.selectbox("Estimated Safety", ["low", "med", "high"])

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Acceptability"):
    # Create input dataframe matching training format
    input_data = pd.DataFrame(
        [[buying, maint, doors, persons, lug_boot, safety]], 
        columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    )
    
    prediction = model.predict(input_data)[0]
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if prediction == 'unacc':
        st.error(f"### Result: Unacceptable (unacc) ❌\nThis car does not meet basic acceptability criteria based on the given features.")
    elif prediction == 'acc':
        st.warning(f"### Result: Acceptable (acc) ✅\nThis car meets basic acceptability standards.")
    elif prediction == 'good':
        st.success(f"### Result: Good (good) 🌟\nThis is a good choice for a car!")
    elif prediction == 'vgood':
        st.success(f"### Result: Very Good (vgood) 🏆\nThis car represents an excellent choice with high standards across the board!")

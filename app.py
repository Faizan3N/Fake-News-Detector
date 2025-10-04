import streamlit as st
import joblib
import warnings
import os

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Check if model files exist
@st.cache_resource
def load_model():
    try:
        # Check if files exist
        if not os.path.exists("vectorizer.jb"):
            raise FileNotFoundError("vectorizer.jb not found")
        if not os.path.exists("lr_model.jb"):
            raise FileNotFoundError("lr_model.jb not found")
            
        vectorizer = joblib.load("vectorizer.jb")
        model = joblib.load("lr_model.jb")
        return vectorizer, model, True
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, False

# Load model with caching
vectorizer, model, model_loaded = load_model()

st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real. ")

inputn = st.text_area("News Article:","")

if st.button("Check News"):
    if not model_loaded or vectorizer is None or model is None:
        st.error("Model is not loaded. Please check the model files.")
    elif inputn.strip():
        try:
            transform_input = vectorizer.transform([inputn])
            prediction = model.predict(transform_input)

            if prediction[0] == 1:
                st.success("The News is Real! ")
            else:
                st.error("The News is Fake! ")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.warning("Please enter some text to Analyze. ") 
    
  

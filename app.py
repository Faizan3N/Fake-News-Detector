import streamlit as st
import joblib
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model files: {e}")
    model_loaded = False

st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real. ")

inputn = st.text_area("News Article:","")

if st.button("Check News"):
    if not model_loaded:
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
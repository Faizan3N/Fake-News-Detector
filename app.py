import streamlit as st
import joblib
import warnings
import numpy as np

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .result-container {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    .real-news {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    
    .fake-news {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        height: 20px;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load models
try:
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model files: {e}")
    model_loaded = False

# Header
st.markdown('<h1 class="main-header">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered News Verification Tool</p>', unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
    <strong>📋 How it works:</strong> Simply paste a news article or text below, and our AI model will analyze it to determine if it's likely to be real or fake news. The model uses advanced machine learning techniques to identify patterns and characteristics of fake news.
</div>
""", unsafe_allow_html=True)

# Main input area
st.markdown("### 📰 Enter News Article")
inputn = st.text_area(
    "Paste your news article here:",
    placeholder="Enter the news article text you want to verify...",
    height=200,
    help="The more text you provide, the more accurate the analysis will be."
)

# Example articles
with st.expander("💡 Need an example? Click here to see sample articles"):
    st.markdown("**Sample Real News:**")
    st.code("Scientists at MIT have developed a new AI system that can detect early signs of Alzheimer's disease through speech patterns. The system achieved 90% accuracy in preliminary tests.")
    
    st.markdown("**Sample Fake News:**")
    st.code("BREAKING: Scientists discover that drinking 10 cups of coffee daily cures all diseases instantly! This miracle cure has been hidden by Big Pharma for decades!")

# Analysis button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze News", use_container_width=True)

# Results section
if analyze_button:
    if not model_loaded:
        st.error("❌ Model is not loaded. Please check the model files.")
    elif inputn.strip():
        try:
            # Make prediction
            transform_input = vectorizer.transform([inputn])
            prediction = model.predict(transform_input)
            
            # Get prediction probability for confidence score
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(transform_input)
                confidence = max(probabilities[0]) * 100
            else:
                confidence = 85  # Default confidence if predict_proba not available
            
            # Display results
            if prediction[0] == 1:
                st.markdown(f"""
                <div class="result-container real-news">
                    ✅ <strong>REAL NEWS</strong><br>
                    This article appears to be legitimate news content.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-container fake-news">
                    ⚠️ <strong>FAKE NEWS</strong><br>
                    This article shows characteristics of fake or misleading content.
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence score
            st.markdown("### 📊 Confidence Score")
            confidence_color = "#28a745" if prediction[0] == 1 else "#dc3545"
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence}%; background-color: {confidence_color};"></div>
            </div>
            <p style="text-align: center; margin-top: 0.5rem;"><strong>{confidence:.1f}%</strong> confidence</p>
            """, unsafe_allow_html=True)
            
            # Additional information
            st.markdown("### ℹ️ Important Notes")
            st.info("""
            - This tool is for educational and research purposes
            - Results should not be the sole basis for determining news authenticity
            - Always verify information through multiple reliable sources
            - Consider the source, date, and context of any news article
            """)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🔬 Powered by Machine Learning | Built with Streamlit</p>
    <p><small>Remember: Always verify news through multiple reliable sources</small></p>
</div>
""", unsafe_allow_html=True) 

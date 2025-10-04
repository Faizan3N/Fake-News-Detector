# Streamlit Cloud Deployment Guide

## Files Required for Deployment

Make sure your repository contains these files:

1. **app.py** - Main Streamlit application
2. **requirements.txt** - Python dependencies
3. **packages.txt** - System packages (optional)
4. **.streamlit/config.toml** - Streamlit configuration
5. **vectorizer.jb** - Trained vectorizer model
6. **lr_model.jb** - Trained logistic regression model

## Deployment Steps

1. **Push to GitHub**: Make sure all files are committed and pushed to your GitHub repository

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set the main file path to `app.py`
   - Click "Deploy"

3. **If deployment fails**:
   - Check the logs in the Streamlit Cloud interface
   - Ensure all model files are present in the repository
   - Verify the requirements.txt format is correct

## Troubleshooting

### ModuleNotFoundError for joblib
- Ensure `requirements.txt` is in the root directory
- Check that the filename is exactly `requirements.txt` (not `requirments.txt`)
- Verify the requirements.txt contains `joblib==1.3.2`

### Model loading errors
- Ensure `vectorizer.jb` and `lr_model.jb` are in the repository root
- Check file permissions and sizes
- Verify the model files are not corrupted

### Version compatibility issues
- The current requirements.txt uses specific versions for compatibility
- If you encounter version conflicts, try updating the versions in requirements.txt

## Current Configuration

- **Streamlit**: 1.23.1
- **Joblib**: 1.3.2  
- **Scikit-learn**: 1.0.2
- **NumPy**: 1.21.6
- **Pandas**: 1.1.5

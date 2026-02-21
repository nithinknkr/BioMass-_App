import streamlit as st
import torch
import numpy as np
from PIL import Image
import pandas as pd
from model_utils import BiomassModel, get_prediction_transforms
import os
import requests

# --- Page Config ---
st.set_page_config(page_title="Pasture Biomass Predictor", layout="centered")

# --- Load Model Function (Cached to save memory) ---
@st.cache_resource
def load_biomass_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiomassModel()

    MODEL_PATH = "Convnext_Tiny_Consistency_model_1.pth"
    MODEL_URL = "https://huggingface.co/Nithin2348/Convnext_Tiny_Consistency_model_1.pth/resolve/main/Convnext_Tiny_Consistency_model_1.pth"

    # 🔽 Download model if not present
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Hugging Face...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded successfully!")

    # Loading model weights with consistency constraints
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# --- Main App Interface ---
st.title("🌱 Pasture Biomass Predictor")
st.write("Upload a pasture image and enter sensor data to predict biomass components.")

# 1. Sidebar for Inputs
st.sidebar.header("Input Parameters")
uploaded_file = st.sidebar.file_uploader("Upload Pasture Image", type=["jpg", "jpeg", "png"])
ndvi_val = st.sidebar.number_input("Pre-GSHH NDVI", min_value=-1.0, max_value=1.0, value=0.5, step=0.01)
height_val = st.sidebar.number_input("Average Height (cm)", min_value=0.0, max_value=200.0, value=15.0, step=0.1)

# 2. Prediction Logic
if st.sidebar.button("Predict Biomass"):
    if uploaded_file is not None:
        model, device = load_biomass_model()
        
        # Process Image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Standardized transformations for ConvNeXt
        transform = get_prediction_transforms()
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Process Metadata (NDVI and Height)
        meta_tensor = torch.tensor([[ndvi_val, height_val]], dtype=torch.float32).to(device)
        
        # Inference
        with torch.no_grad():
            preds_log = model(img_tensor, meta_tensor)
            # Inverse of log1p is expm1
            raw_preds = torch.expm1(preds_log).cpu().numpy()[0]
            
            # --- FIX: Ensure no negative values due to regression noise ---
            preds_grams = np.maximum(0, raw_preds)
        
        # 3. Display Results
        st.success("Analysis Complete!")
        
        # Display Metrics in columns
        labels = ['Green (g)', 'Dead (g)', 'Clover (g)', 'GDM (g)', 'Total (g)']
        cols = st.columns(len(labels))
        for col, label, val in zip(cols, labels, preds_grams):
            col.metric(label, f"{val:.2f}")

        # --- Visualization ---
        st.markdown("---")
        st.subheader("Composition Analysis")
        
        # Chart focusing on the three main forage components
        chart_data = pd.DataFrame({
            'Component': ['Green', 'Dead', 'Clover'],
            'Weight (g)': [preds_grams[0], preds_grams[1], preds_grams[2]]
        })
        
        st.bar_chart(data=chart_data, x='Component', y='Weight (g)', color="#4CAF50")
        
        # --- Summary Insight ---
        st.subheader("Field Insights")
        
        # Calculating ratios for nutritional analysis
        green_ratio = preds_grams[0] / preds_grams[4] if preds_grams[4] > 0 else 0
        
        if green_ratio > 0.6:
            st.info("💡 **Insight:** This area has high green biomass. Ideal for active grazing.")
        elif preds_grams[1] > preds_grams[0]:
            st.warning("💡 **Insight:** Dead material exceeds green growth. Consider checking for water stress or over-maturation.")
        
        # Clover detection for protein monitoring
        if preds_grams[2] > (0.2 * preds_grams[4]):
            st.success("🍀 **Insight:** High Clover content detected! This indicates good nitrogen fixation in the soil.")
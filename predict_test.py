import torch
import numpy as np
from PIL import Image
from model_utils import BiomassModel, get_prediction_transforms
import os
import requests

def run_test():
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    model = BiomassModel()
    # Replace with your actual filename if it's different
    model_path = "Convnext_Tiny_Consistency_model_1.pth"
    model_url = "https://huggingface.co/Nithin2348/Convnext_Tiny_Consistency_model_1.pth/resolve/main/Convnext_Tiny_Consistency_model_1.pth"

    if not os.path.exists(model_path):
        print("Downloading model from Hugging Face...")
    r = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(r.content)
    print("Model downloaded successfully!")         
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Create a Dummy Image & Meta (Simulating a user input)
    # 224x224 RGB Image
    dummy_img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    transform = get_prediction_transforms()
    img_tensor = transform(dummy_img).unsqueeze(0).to(device)

    # Dummy Metadata (NDVI and Height)
    dummy_meta = torch.tensor([[0.5, 15.0]]).to(device) # NDVI=0.5, Height=15cm

    # 4. Predict
    with torch.no_grad():
        preds_log = model(img_tensor, dummy_meta)
        
        # 5. Convert back from Log-scale to Grams (expm1 is e^x - 1)
        preds_grams = torch.expm1(preds_log).cpu().numpy()[0]

    # 6. Display Results
    labels = ['Green_g', 'Dead_g', 'Clover_g', 'GDM_g', 'Total_g']
    print("\n--- Test Prediction (Grams) ---")
    for label, val in zip(labels, preds_grams):
        print(f"{label}: {val:.2f}g")

if __name__ == "__main__":
    run_test()
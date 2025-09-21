# quantize_clip.py (Corrected Version)
import torch
import os
from transformers import CLIPModel

# ======================= Configuration =======================
# Path to the original FP32 model
FP32_MODEL_PATH = "./model/chinese-roberta-wwm-ext" # Make sure this path is correct

# Directory to save the quantized model
QUANTIZED_MODEL_DIR = "./model_quantized"
# Filename for the quantized model
QUANTIZED_MODEL_FILENAME = "quantized_clip.pth"
# =============================================================

# Ensure the save directory exists
if not os.path.exists(QUANTIZED_MODEL_DIR):
    os.makedirs(QUANTIZED_MODEL_DIR)

# Full path for the output file
output_file_path = os.path.join(QUANTIZED_MODEL_DIR, QUANTIZED_MODEL_FILENAME)

print("--- Loading FP32 model ---")
# Quantization must be done on the CPU
model_fp32 = CLIPModel.from_pretrained(FP32_MODEL_PATH)
model_fp32.eval() # Must be in evaluation mode

print("--- Applying dynamic quantization (INT8) to sub-modules ---")

# Quantize the text and vision sub-models separately
# We will quantize all linear layers (torch.nn.Linear)
quantized_text_model = torch.quantization.quantize_dynamic(
    model_fp32.text_model, {torch.nn.Linear}, dtype=torch.qint8
)
quantized_vision_model = torch.quantization.quantize_dynamic(
    model_fp32.vision_model, {torch.nn.Linear}, dtype=torch.qint8
)

# Place the quantized sub-modules back into the main model structure
model_fp32.text_model = quantized_text_model
model_fp32.vision_model = quantized_vision_model

print(f"--- Saving quantized model using torch.save() to '{output_file_path}' ---")
# Use PyTorch's native torch.save to save the entire model object
torch.save(model_fp32, output_file_path)

print("\n--- Quantization and saving complete! ---")
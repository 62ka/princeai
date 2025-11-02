import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- CONFIG ---
MODEL_NAME = "bigcode/starcoder"  # StarCoder model
SAVE_DIR = Path("saved_code")
SAVE_DIR.mkdir(exist_ok=True)

# --- LOAD MODEL ---
print("Loading model... this may take a while depending on your hardware.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}.")

# --- FUNCTION TO GENERATE CODE ---
def generate_code(prompt, max_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return code

# --- INTERACTIVE PROMPT ---
while True:
    user_prompt = input("\nEnter coding task (or 'exit' to quit):\n> ")
    if user_prompt.lower() in ["exit", "quit"]:
        break

    code = generate_code(user_prompt)
    print("\n--- Generated Code ---\n")
    print(code)
    
    # Save code to file
    file_name = SAVE_DIR / f"code_{len(os.listdir(SAVE_DIR))+1}.txt"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"\nSaved generated code to {file_name}")

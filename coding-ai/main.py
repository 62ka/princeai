import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- CONFIG ---
MODEL_NAME = "bigcode/starcoder-small"  # Public, smaller version
SAVE_DIR = Path("saved_code")
SAVE_DIR.mkdir(exist_ok=True)

# --- LOAD MODEL ---
print("Downloading and loading model... this may take a few minutes on first run.")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
model.to(device)
model.eval()
print("Model loaded successfully!")

# --- FUNCTION TO GENERATE CODE ---
def generate_code(prompt, max_tokens=512):
    print("Generating code...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return code

# --- INTERACTIVE PROMPT ---
print("\nPrinceAI is ready! Type your coding task or 'exit' to quit.")
while True:
    user_prompt = input("\n> ")
    if user_prompt.lower() in ["exit", "quit"]:
        print("Exiting PrinceAI. Goodbye!")
        break

    code = generate_code(user_prompt)
    print("\n--- Generated Code ---\n")
    print(code)
    
    # Save code to file
    file_name = SAVE_DIR / f"code_{len(os.listdir(SAVE_DIR))+1}.txt"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"\nSaved generated code to {file_name}")

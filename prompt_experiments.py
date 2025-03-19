
from transformers import pipeline
import os
from huggingface_hub import login

# Securely ask for API key at runtime (no hardcoded secret)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or input("Enter your Hugging Face API Key: ")
login(token=HUGGINGFACE_API_KEY)

# Load a free AI chatbot model
chatbot = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cuda")

# Function to generate AI responses
# Define a test prompt
test_prompt = "How can AI improve productivity in the workplace?"

# Tokenize the input
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

# Generate response
with torch.no_grad():
    output = model.generate(
        **inputs, 
        max_new_tokens=100,  # Limit output length
        do_sample=True,  # Enable sampling for diverse responses
        temperature=0.7,  # Adjust temperature for randomness
        top_p=0.9  # Nucleus sampling for better quality responses
    )

# Decode and print the model's response
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("🔹 Model Response:", generated_text)
for prompt in prompts:
    print("\n🔹 Prompt:", prompt)
    print("💡 AI Response:", generate_response(prompt))

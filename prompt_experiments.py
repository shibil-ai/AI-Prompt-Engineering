
from transformers import pipeline
import os
from huggingface_hub import login

# Securely ask for API key at runtime (no hardcoded secret)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or input("Enter your Hugging Face API Key: ")
login(token=HUGGINGFACE_API_KEY)

# Load a free AI chatbot model
chatbot = pipeline("text-generation", model="facebook/opt-1.3b")

# Function to generate AI responses
def generate_response(prompt):
    input_prompt = f"### Question: {prompt}
### Answer:"
    
    response = chatbot(
        input_prompt,
        max_length=150,  # Keep response length balanced
        truncation=True,  
        pad_token_id=50256,  
        num_return_sequences=1,  
        temperature=0.7,  # Add randomness for more creativity
        top_p=0.9,  # Use nucleus sampling for better diversity
        do_sample=True,  # Enable sampling for non-repetitive responses
    )
    return response[0]["generated_text"].replace(input_prompt, "").strip()

# Test with different prompts
prompts = [
    "Explain quantum physics in simple terms.",
    "Give me 5 startup ideas related to AI.",
    "Write a creative story about a robot in space."
]

for prompt in prompts:
    print("\n🔹 Prompt:", prompt)
    print("💡 AI Response:", generate_response(prompt))

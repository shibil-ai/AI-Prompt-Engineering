
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
    response = chatbot(
        prompt,  # Directly passing the prompt
        max_length=150,  
        truncation=True,  
        pad_token_id=50256,  
        num_return_sequences=1,  
        temperature=0.8,  # More creativity
        top_p=0.95,  # More diversity in responses
        do_sample=True  # Ensures varied output
    )

    return response[0]["generated_text"]
# Test with different prompts
prompts = [
    "Explain quantum physics in simple terms.",
    "Give me 5 startup ideas related to AI.",
    "Write a creative story about a robot in space."
]

for prompt in prompts:
    print("\n🔹 Prompt:", prompt)
    print("💡 AI Response:", generate_response(prompt))

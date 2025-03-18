
from transformers import pipeline
import os
from huggingface_hub import login

# Securely ask for API key at runtime (no hardcoded secret)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or input("Enter your Hugging Face API Key: ")
login(token=HUGGINGFACE_API_KEY)

# Load a free AI chatbot model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Function to generate AI responses
def generate_response(prompt):
    response = chatbot(prompt, max_length=100)[0]["generated_text"]
    return response

# Test with different prompts
prompts = [
    "Explain quantum physics in simple terms.",
    "Give me 5 startup ideas related to AI.",
    "Write a creative story about a robot in space."
]

for prompt in prompts:
    print("\n🔹 Prompt:", prompt)
    print("💡 AI Response:", generate_response(prompt))

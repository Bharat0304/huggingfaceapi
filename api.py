from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face API key from environment variables
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")

# Initialize HuggingFace endpoint (example: meta-llama model)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  # Replace with your model
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    temperature=0.7,
    max_new_tokens=256
)

# Wrap it in a ChatHuggingFace instance
model = ChatHuggingFace(llm=llm)

# Invoke the model
response = model.invoke("What is the capital of India?")
print(response.content)

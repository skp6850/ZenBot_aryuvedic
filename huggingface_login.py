import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Get token from environment variable
token = os.getenv('HUGGINGFACE_TOKEN')
login(token)

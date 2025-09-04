# Task 1 script

import os
from dotenv import load_dotenv
load_dotenv()

# Environment Variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

print("Azure OpenAI API Key:", api_key)
print("Azure OpenAI Endpoint:", endpoint)
print("Azure OpenAI Deployment:", deployment)
print("Azure OpenAI API Version:", api_version)
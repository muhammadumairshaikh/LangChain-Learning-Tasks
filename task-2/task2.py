# Script Task 2

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback

# Load env vars
load_dotenv()

# Load credentials
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Configure Azure OpenAI LLM
llm = AzureChatOpenAI(
    openai_api_key=api_key,
    azure_endpoint=endpoint,
    deployment_name=deployment,
    openai_api_version=api_version,
    temperature=0.1  
)

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text into exactly 3 sentences:\n\n{text}"
)

# Create the chain
summarization_chain = LLMChain(llm=llm, prompt=prompt_template)

input_text = """
Artificial intelligence (AI) is a rapidly advancing field of computer science that focuses on the creation of systems capable of performing tasks that normally require human intelligence. These tasks include problem-solving, natural language understanding, perception, and decision-making. The progress in AI has been driven by advances in machine learning, particularly deep learning, which leverages large datasets and powerful computing resources to train complex models. One of the most notable applications of AI is in natural language processing, where models such as chatbots and virtual assistants are capable of understanding and responding to human language with increasing sophistication. AI also plays a significant role in computer vision, enabling systems to recognize and categorize images, detect objects, and even interpret medical scans. Additionally, AI has been integrated into recommendation engines, predictive analytics, robotics, and autonomous vehicles, transforming multiple industries. Despite its benefits, AI also raises ethical concerns, including bias in algorithms, job displacement, and questions about accountability. Governments, researchers, and organizations are actively exploring frameworks to ensure AI is developed responsibly. The potential of AI is vast, and while it offers opportunities for innovation and efficiency, it also requires careful oversight to balance progress with societal impact.
"""

# Run the chain
summary = summarization_chain.run({"text": input_text})
print("3-sentence summary:\n", summary)

print("------------------------------")

# 1 sentence Summary
prompt_template_1sentence = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text into exactly 1 sentence:\n\n{text}"
)

one_sentence_chain = LLMChain(llm=llm, prompt=prompt_template_1sentence)

summary_one = one_sentence_chain.run({"text": input_text})
print("\n1-sentence summary:\n", summary_one)

# Tokens Summary
# 3-sentence summary with token tracking
with get_openai_callback() as cb:
    summary = summarization_chain.run({"text": input_text})

    # Print token stats
    print("\n--- Token Usage (3-sentence) ---")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")

# 1-sentence summary with token tracking
with get_openai_callback() as cb:
    summary_one = one_sentence_chain.run({"text": input_text})

    # Print token stats
    print("\n--- Token Usage (1-sentence) ---")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")

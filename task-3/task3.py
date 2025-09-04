# Script Task#3

import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


# Load env
load_dotenv()

# Loading the file
loader = TextLoader("ai_intro.txt")
documents = loader.load()

# Splitting into Chunks
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_documents(documents)

# Embeddings and Vector Store
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",  # Azure deployment for embeddings
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"), 
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

vectorstore = FAISS.from_documents(docs, embeddings)

# Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 5. Query retriever
query = "AI milestones"
retrieved_docs = retriever.get_relevant_documents(query)
retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

# 6. Reuse summarization chain from Task 2
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text into 3 sentences:\n\n{text}"
)

summarization_chain = LLMChain(llm=llm, prompt=prompt_template)

# 7. Generate summary
summary = summarization_chain.run({"text": retrieved_text})
print("\nSummary of AI milestones:\n", summary)


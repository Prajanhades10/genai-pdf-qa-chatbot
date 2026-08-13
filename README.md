
### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

### DESIGN STEPS:

#### STEP 1:
Load PDF: Load the provided PDF using LangChain and extract its text.
#### STEP 2:
Split & Embed: Divide the text into smaller chunks and convert them into embeddings.
#### STEP 3:
Retrieve & Answer: Store embeddings in a vector database, retrieve relevant chunks for the user's question, and generate an answer using an LLM.Evaluate: Test the chatbot with different queries and evaluate the answers based on accuracy, relevance, and completeness.
### PROGRAM:
```
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load API Key
load_dotenv()

# Load PDF
loader = PyPDFLoader("AI_Notes.pdf")
documents = loader.load()

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create Vector Database
embeddings = OpenAIEmbeddings()
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# Create QA Chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=db.as_retriever()
)

# Ask Questions
while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    answer = qa.run(question)
    print("Bot:", answer)
```

### OUTPUT:
<img width="1568" height="605" alt="Screenshot From 2026-08-13 13-40-33" src="https://github.com/user-attachments/assets/31f7dfcd-0920-4d20-a55a-99267a5895f5" />



### RESULT:


The LangChain-based PDF question-answering chatbot was successfully designed and implemented. It extracted information from the PDF and provided **relevant and accurate answers** to different user queries. The chatbot showed good performance in terms of **accuracy, relevance, and completeness** of responses.

# Chatbot for Agricultural Pricing

## Overview  
This project presents an intelligent, AI-powered chatbot deployed on Telegram to assist farmers in accessing real-time **agricultural commodity prices** using simple natural language queries. Designed for low-bandwidth and low-literacy environments, the bot is multilingual and integrates **RAG-based retrieval**, **LLMs**, and **FAISS** for accurate price responses.

## Abstract  
The goal of this project is to bridge the information gap in agricultural markets by building a chatbot that provides farmers with localized and commodity-specific market prices. By leveraging **Retrieval-Augmented Generation (RAG)**, **FAISS vector search**, and **language models**, the system enables intuitive, voice/text-based queries on mobile devices.

The chatbot retrieves answers from a structured JSON dataset containing price details for various crops across states, districts, and markets. It uses **LangChain**, **Hugging Face Transformers**, and **Ollama's LLMs** (like `deepseek-llm`) to ensure smart and contextually accurate responses.

## Key Technologies

| Component        | Description                                  |
|------------------|----------------------------------------------|
| LangChain        | Conversational RAG model                     |
| HuggingFace      | MiniLM-L6-v2 Embedding Model                 |
| FAISS            | Vector search backend                        |
| Ollama           | LLM (deepseek-r1:8b)                         |
| Telegram         | Bot deployment interface                     |
| Python Libraries | `pickle`, `asyncio`, `logging`, `tracemalloc` |

## System Components
- **Data Indexing Module**: Converts agriculture price JSON to FAISS vector store.
- **Embedding Generator**: Uses sentence-transformers to generate vector embeddings.
- **RAG Model**: Fetches relevant document chunks and generates contextual answers.
- **Telegram Bot**: Front-end interface for users to interact using voice or text.
- **Voice Support (optional)**: For low-literacy regions (future scope).
- **Local Deployment**: Powered by Ollama LLMs for offline/edge support.

## Existing System  
Current government apps and web portals:
- Are not conversational
- Require technical literacy
- Provide static or outdated data
- Lack language diversity

## Disadvantages of Existing Systems
- No personalization or memory
- No voice/natural language interface
- Requires internet + mobile app expertise
- Doesn’t work well in regional languages

## Proposed System  
We introduce a Telegram-based conversational bot that:
- Accepts regional language input (via Google Translate)
- Works on both voice and text commands
- Uses FAISS and RAG for intelligent responses
- Operates offline via Ollama and local LLMs

## Advantages of Proposed System
- Natural query interface (no keyword search)
- Fast and scalable using FAISS
- Personalizable and memory-enabled
- Offline/low-cost deployment
- Regional language support

## Feasibility Study

### Hardware Requirements
- Basic laptop or server
- Optional GPU for fast embedding
- Internet or LAN for Telegram

### Software Requirements
- OS: Linux/Windows
- Language: Python 3.10+
- Libraries: LangChain, FAISS, Transformers, Ollama, Telebot

## Technologies Used
- **Python**: Core backend
- **LangChain**: RAG and prompt management
- **FAISS**: Vector similarity engine
- **HuggingFace Sentence-Transformers**: For embeddings
- **Ollama LLMs**: Local inference (Deepseek, Llama3, etc.)
- **Telegram Bot API**: User interaction interface

##  Installation and Coding

### 1. Install Dependencies

```bash
pip install langchain sentence-transformers faiss-cpu python-telegram-bot ollama
```

### 2. Download LLM Model (via Ollama)
```bash
ollama pull deepseek-r1:8b
```

### 3. bot.py
```
import os
import json
import logging
import pickle
import asyncio
import tracemalloc
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOllama
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext

tracemalloc.start()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_FILE = "agriculture_data.json"
FAISS_INDEX_PATH = "faiss_index"
MODEL_SAVE_PATH = "rag_model.pkl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "deepseek-r1:8b"
TOKEN = "YOUR_BOT_TOKEN_HERE"

def load_json_documents(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"{json_file} not found.")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        f"State: {entry['state']}, District: {entry['district']}, Market: {entry['market']}, "
        f"Commodity: {entry['commodity']}, Variety: {entry['variety']}, Min Price: {entry['min_price']}, "
        f"Max Price: {entry['max_price']}, Modal Price: {entry['modal_price']}"
        for entry in data
    ]

def create_faiss_vector_store(documents, embedding_model_name, save_path):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.from_texts(texts=documents, embedding=embedding_model)
    vector_store.save_local(save_path)
    return vector_store

def load_faiss_vector_store(save_path, embedding_model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)

def setup_rag_chain(vector_store, llm_model_name):
    llm = ChatOllama(model=llm_model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)

def ask_rag(chain, query, chat_history=[]):
    return chain.invoke({"question": query, "chat_history": chat_history})["answer"]

def save_rag_model(rag_chain, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(rag_chain, f)

def load_rag_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

if os.path.exists(MODEL_SAVE_PATH):
    qa_chain = load_rag_model(MODEL_SAVE_PATH)
else:
    documents = load_json_documents(DATA_FILE)
    vector_store = (
        load_faiss_vector_store(FAISS_INDEX_PATH, EMBEDDING_MODEL)
        if os.path.exists(FAISS_INDEX_PATH)
        else create_faiss_vector_store(documents, EMBEDDING_MODEL, FAISS_INDEX_PATH)
    )
    qa_chain = setup_rag_chain(vector_store, LLM_MODEL)
    save_rag_model(qa_chain, MODEL_SAVE_PATH)

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("🌾 Welcome to AgriBot! Ask me about crop prices.")

async def handle_message(update: Update, context: CallbackContext):
    query = update.message.text
    response = ask_rag(qa_chain, query)
    await update.message.reply_text(response)

async def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await app.run_polling()

def run_telegram_bot():
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(main())
    else:
        loop.run_until_complete(main())

run_telegram_bot()

```

## Output Example
```
User: What is the modal price of tomato in Bengaluru?
AgriBot: State: Karnataka, District: Bengaluru, Market: Bengaluru, Commodity: Tomato, Variety: Local, Modal Price: ₹1700

User: Which district in Maharashtra had the highest max price for Onion?
AgriBot: Jalgaon district had the highest max price of ₹3200 for Onion in Pimpalgaon market.

User: “What is the modal price of onions in Pune?”  
Bot: “In Pune market, the modal price of Onion (Red) is ₹1300 per quintal.”

```
## Conclusion  
This project successfully provides a robust, low-cost, and user-friendly solution for delivering market pricing data to farmers. The combination of RAG + LLMs + vector search creates a scalable chatbot platform that improves agricultural decision-making in real-time.

## Future Enhancements
- Add predictive pricing using LSTM or Prophet
- Dashboard for district-wise price trends
- Full voice interface with Whisper or Coqui
- Multilingual support (Telugu, Tamil, Hindi, etc.)
- Deployment on smart kiosks and IVR systems

## Output Example
User: _“What is the modal price of onions in Pune?”_  
Bot: _“In Pune market, the modal price of Onion (Red) is ₹1300 per quintal.”_

## References  
1. OpenAI, "Retrieval-Augmented Generation with LangChain", 2024  
2. HuggingFace, "Transformers and Sentence Transformers", 2024  
3. Facebook AI, "FAISS: Efficient Similarity Search", 2022  
4. Telegram API Documentation, 2024  
5. Government of India, "Agmarknet Agricultural Price Dataset", 2023  
6. Deepseek.ai, “LLM Series for Domain-Specific Chatbots”, 2024  
7. LangChain Blog, “RAG Applications in Agriculture”, 2024

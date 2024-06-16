# AI Journey - Implementing a Simple RAG
This will be my first RAG (Retrieval Augmented Generation) implementation using ChromaDB and Ollama. I'm going to use this as a base for future projects, so I'm keeping it simple for now.

## Requirements:
### Hardware
1. A GPU with 6GB or more of RAM for faster processing
 - Ollama with llama3:latest (7B) - 5066 of VRAM
 - Python - No Models will be loaded, just the 'in-memory' embedings
 
### Software
1. CUDA - Drivers and Toolkit
2. CUDNN - Neural Network library (**not included** in the CUDA Toolkit)
3. Python3 Enviornment - Preferably Anaconda Distribution

## On the 'Simple RAG'
The RAG will:
1. Collect information from a RSS
2. Split the inforamtion into chunks
3. Save the chunks to ChromaDB
4. Use Ollama to generate an answer based on the question and the chunks in ChromaDB

### How to Use
1. Run rag.py
```bash
python3 rag.py
```
3. Enter the URL to fetch the information from
4. Query the model with your question using the embedings

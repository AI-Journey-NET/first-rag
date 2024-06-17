import numpy as np
import requests

LLM_GENERATE_API="http://localhost:11434/api/generate"
LLM_GENERATE_MODEL="llama3"
LLM_EMBEDINGS_API="http://localhost:11434/api/embeddings"
LLM_EMBEDINGS_MODEL="nomic-embed-text"
LLM_HEADERS = {
    'Content-Type': 'application/json'
}

def fetch_content(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    return response.text

def split_into_chunks(text, max_length=2048):
    words = text.split()
    chunks = []
    chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # Adding 1 for the space
        if current_length + word_length <= max_length:
            chunk.append(word)
            current_length += word_length
        else:
            chunks.append(' '.join(chunk))
            chunk = [word]
            current_length = word_length

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        response = requests.post(
            LLM_EMBEDINGS_API,
            headers=LLM_HEADERS,
            json={
                'prompt': chunk,
                'model': LLM_EMBEDINGS_MODEL  # Ensure the model is suitable for embedding generation
            }
        )
        response.raise_for_status()
        embedding = response.json()['embedding']
        embeddings.append((embedding, chunk))  # Store tuple of embedding and associated prompt
    return embeddings

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_relevant_embeddings(embeddings, prompt_embedding, top_n=5):
    similarities = [(cosine_similarity(prompt_embedding, embedding), chunk) for embedding, chunk in embeddings]
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_similarities = similarities[:top_n]
    return top_similarities

def generate_response(relevant_data, user_input):
    response = requests.post(
        LLM_GENERATE_API,
        headers=LLM_HEADERS,
        json={
            'model': LLM_GENERATE_MODEL,  # Ensure the model is suitable for generating responses
            'stream': False,
            'prompt': f"Using these data: {', '.join([chunk for _, chunk in relevant_data])}. Respond to this prompt: {user_input}",
        }
    )
    response.raise_for_status()
    return response.json()['response']

def main():
    print("Fetching data")
    url = input("Enter the URL to fetch content from: ")
    content = fetch_content(url)

    print("Splitting chunks")
    text_chunks = split_into_chunks(content)
    print("Generating embedings")
    embeddings = generate_embeddings(text_chunks)

    print("Embeddings generated and stored in memory.")

    while True:
        user_input = input("Enter your input: ")

        # Generate the embedding for the user input
        response = requests.post(
            LLM_EMBEDINGS_API,
            headers=LLM_HEADERS,
            json={
                'prompt': user_input,
                'model': LLM_EMBEDINGS_MODEL
            }
        )
        response.raise_for_status()
        prompt_embedding = response.json()['embedding']

        top_relevant_embeddings = get_relevant_embeddings(embeddings, prompt_embedding, top_n=5)
        ai_response = generate_response(top_relevant_embeddings, user_input)
        
        print("AI Response:", ai_response)

if __name__ == "__main__":
    main()
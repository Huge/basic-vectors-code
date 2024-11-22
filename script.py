import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Read the text from the file
def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Step 2: Chunk the text into smaller parts
def chunk_text(text, chunk_size=512, overlap=0):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        if end >= len(words):
            chunks.append(' '.join(words[start:]))
            break
        else:
            chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# Step 3: Generate embeddings for each chunk
def generate_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

# Step 5: Compute cosine similarity and find the 2 most similar chunks
def find_most_similar(embeddings, top_k=2):
    similarities = cosine_similarity(embeddings)
    np.fill_diagonal(similarities, -1)  # Ignore self-similarity
    most_similar_indices = np.argsort(-similarities, axis=1)[:, :top_k]
    return most_similar_indices

# Main function
def main(file_path, chunk_size=512, overlap=0, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=2):
    # Read the text
    text = read_text(file_path)
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"Number of chunks: {len(chunks)}")
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks, model_name=model_name)
    
    # Print each embedding
    for i, embedding in enumerate(embeddings):
        print(f"Embedding for chunk {i+1}:")
        print(embedding.numpy())
        print()
    
    # Find the 2 most similar chunks for each chunk
    most_similar_indices = find_most_similar(embeddings.numpy(), top_k=top_k)
    
    # Print the 2 most similar chunks for each chunk
    for i, indices in enumerate(most_similar_indices):
        print(f"Chunk {i+1} is most similar to chunks:", end=' ')
        for idx in indices:
            print(f"{idx+1}", end=' ')
        print()
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Please provide the path to the text file.")
        sys.exit(1)
    file_path = sys.argv[1]
    #main(file_path)
    ## All those are (Embedding Dimensions, Max Tokens Average) 384, 512
    main(file_path, model_name="avsolatorio/NoInstruct-small-Embedding-v0")#"avsolatorio/GIST-small-Embedding-v0") # or perhaps less popular  abhinand/MedEmbed-small-v0.1
    ## jinaai/jina-embeddings-v2-small-en is 512 numbers from up to 8192 context, which seems great

from mem0 import Memory

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test_memory",
            "path": "./qdrant_data",  # saved locally on disk, right here
            "embedding_model_dims": 768,  # must match nomic-embed-text's output size
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen2.5:0.5b",
            "ollama_base_url": "http://localhost:11434",  # Ollama's local server address
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
            "embedding_dims": 768,  # nomic-embed-text outputs 768-length vectors
        },
    },
}

m = Memory.from_config(config)

# Store one memory
m.add("My name is Ayesha and I have a 4GB RAM Linux laptop.", user_id="ayesha")

# Retrieve it back
results = m.search("what do you know about my laptop?", filters={"user_id": "ayesha"})

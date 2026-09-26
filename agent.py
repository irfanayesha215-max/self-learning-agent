from mem0 import Memory
import ollama

MODEL = "qwen2.5:1.5b"  # bumped up from 0.5b for reliable structured extraction

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "agent_memory",
            "path": "./qdrant_data",
            "embedding_model_dims": 768,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": MODEL,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
            "embedding_dims": 768,
        },
    },
}

m = Memory.from_config(config)
USER_ID = "ayesha"


def chat(user_message: str) -> str:
    hits = m.search(user_message, filters={"user_id": USER_ID}, limit=5)
    relevant_facts = [h["memory"] for h in hits.get("results", [])]

    memory_block = "\n".join(f"- {fact}" for fact in relevant_facts) or "(nothing known yet)"

    system_prompt = f"""You are a helpful assistant with memory of past conversations.
Known facts about the user:
{memory_block}

Answer briefly and naturally. Use the facts above only if relevant."""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    reply = response["message"]["content"]

    m.add(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ],
        user_id=USER_ID,
    )

    return reply


if __name__ == "__main__":
    print("Agent ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        print(f"\nAgent: {chat(user_input)}\n")

    print("\n--- What the agent has learned about you ---")
    all_memories = m.get_all(filters={"user_id": USER_ID})
    for mem in all_memories.get("results", []):
        print("-", mem["memory"])

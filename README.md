# Self-Learning Agent (mem0 + Ollama)

A personal AI agent that remembers facts about the user across sessions, built entirely with free, local, open-source tools — no paid APIs, no cloud dependencies.

## How it works

Most chatbots forget everything the moment a session ends. This agent doesn't, because it doesn't rely on the LLM's own memory at all — it uses an external memory layer (mem0) that persists facts to disk and retrieves them on every new message.

The loop, on every message:

1. **Search** — the user's message is embedded into a vector and used to search a local vector database for relevant past facts
2. **Inject** — those facts are added to the system prompt as context
3. **Generate** — the LLM responds, now "aware" of relevant history
4. **Extract & store** — the exchange is passed back to mem0, which extracts durable facts (adding, updating, or discarding them against what's already known) and saves them to disk

This means the agent's knowledge of the user grows over time, purely through retrieval — no fine-tuning or model training involved.

## Tech stack

| Component | Tool | Purpose |
|---|---|---|
| LLM (chat + extraction) | [Ollama](https://ollama.com) running `qwen2.5:1.5b` | Generates replies and extracts structured facts from conversations |
| Embeddings | Ollama running `nomic-embed-text` | Converts text to 768-dimension vectors for meaning-based search |
| Memory layer | [mem0](https://github.com/mem0ai/mem0) | Manages fact extraction, storage, retrieval, and conflict resolution |
| Vector store | [Qdrant](https://qdrant.tech) (local, on-disk) | Stores and searches memory vectors |

Everything runs on-device. No API keys, no billing, no internet required after initial setup.

## Setup

**Requirements:** Python 3.12+, [Ollama](https://ollama.com/download) installed.

```bash
# Pull the required models
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text

# Set up the Python environment
python3 -m venv agent_env
source agent_env/bin/activate
pip install mem0ai qdrant-client ollama
```

## Usage

```bash
python agent.py
```

Chat normally. Type `quit` to exit — this also prints every fact the agent has learned about you so far.

Facts persist in `./qdrant_data` between runs, so close the program and reopen it later — it'll still remember you.

## Why this design

- **Fully local:** every other tutorial online defaults to OpenAI/Anthropic APIs for both the LLM and embeddings. This version proves the same architecture works with zero-cost, self-hosted models — better for learning what's actually happening under the hood, and usable with no budget.
- **Small models on purpose:** `qwen2.5:1.5b` was chosen after testing showed `qwen2.5:0.5b` was too small to reliably follow mem0's structured-extraction prompts (it would silently fail to store any memories). This is a real, documented trade-off of running fully local on modest hardware.

## Possible extensions

- Let the user ask "what do you know about me?" directly inside the chat
- Add a command to forget/delete a specific memory
- Give the agent a narrower purpose (e.g. a study tracker, journaling companion) instead of general chat
- Swap in a larger local model if running on more capable hardware

## Built while learning

This project was built step-by-step to understand embeddings, vector search, and agent memory architecture from first principles — not just to get a working demo.

# CrustData chatbot

Fast and accurate chatbot for CrustData's documentation.

## Setup and Installation

### 1. Clone the repository:

```bash
git clone https://github.com/AkashKarnatak/crustdata-chatbot.git
```

### 2. Navigate to the project directory:

```bash
cd crustdata-chatbot
```

### 3. Setup environment and install dependencies:

```bash
cd markdown-chunker
pnpm i
cd ..
uv sync
export GEMINI_API_KEY=<YOUR-GEMINI-API-KEY>
```

### 4. Run database:

Setup [QDrant locally](https://qdrant.tech/documentation/quickstart/#how-to-get-started-with-qdrant-locally)

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### 5. Seed vector database:

Before we can chat with the chatbot we need to seed our vector database.

```bash
uv run generate_embeddings.py
```

### 6. Launch UI:

Now we can launch the UI to chat with the chatbot.

```bash
uv run streamlit run ui.py
```

## License

This project is not open-source. DO NOT COPY MY CODE.

![my code](https://i.imgflip.com/9gf6hs.jpg)

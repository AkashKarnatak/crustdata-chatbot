import os
import tqdm
import asyncio
import subprocess
from glob import glob
from generate_context import generate_summary
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding


def embed_documents():
    client = QdrantClient("http://localhost:6333")
    if client.collection_exists("crustdata"):
        print("\nCollection already exist!\n")
        return

    all_mini = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    bm25 = SparseTextEmbedding("Qdrant/bm25")
    colbert = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

    document_template = """<summary>
{0}
</summary>

{1}
"""

    client.create_collection(
        "crustdata",
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )
    print("\nCollection created successfully!\n")

    sources = []
    documents = []
    files = glob("./docs/*.md")
    for path in files:
        stem = os.path.basename(os.path.splitext(path)[0])
        chunk_paths = sorted(
            filter(lambda x: not os.path.isdir(x), glob(f"./docs/chunks/{stem}/*"))
        )
        for chunk_path in chunk_paths:
            chunk_name = os.path.basename(chunk_path)
            summary_path = os.path.join(f"./docs/chunks/{stem}/summary", chunk_name)
            with open(chunk_path, "r") as f:
                chunk = f.read()
            with open(summary_path, "r") as f:
                summary = f.read()

            document = document_template.format(summary.strip(), chunk.strip())
            sources.append(f"{stem}-{chunk_name}")
            documents.append(document)

    batch_size = 4
    for i in tqdm.trange(0, len(documents), batch_size):
        source_batch = sources[i : i + batch_size]
        doc_batch = documents[i : i + batch_size]
        dense_embds = list(all_mini.passage_embed(doc_batch))
        sparse_embds = list(bm25.passage_embed(doc_batch))
        late_embds = list(colbert.passage_embed(doc_batch))

        client.upload_points(
            "crustdata",
            points=[
                models.PointStruct(
                    id=int(i + j),
                    vector={
                        "all-MiniLM-L6-v2": dense_embds[j].tolist(),
                        "bm25": sparse_embds[j].as_object(),
                        "colbertv2.0": late_embds[j].tolist(),
                    },
                    payload={
                        "source": source_batch[j],
                        "doc": doc_batch[j],
                    },
                )
                for j in range(len(doc_batch))
            ],
            batch_size=len(doc_batch),
        )


def chunk_docs():
    try:
        result = subprocess.run(
            ["node", "./markdown-chunker/main.js"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("STDOUT:\n", result.stdout, sep="")
        print("STDERR:\n", result.stderr, sep="")

    except subprocess.CalledProcessError as e:
        print(f"Error while chunking documentation files: {e}")
        print("STDOUT:\n", e.stdout, sep="")
        print("STDERR:\n", e.stderr, sep="")
        exit(1)


async def async_main():
    await generate_summary()


def main():
    print("\nChunking documents...\n")
    chunk_docs()
    print("\nDocs chunked successfully!\n")
    print("\nGenerating contextual summaries for each chunk...\n")
    asyncio.run(async_main())
    print("\nSummaries generated successfully\n")
    print("\nGenerating embeddings...\n")
    embed_documents()
    print("\nEmbeddings generated successfully!\n")


if __name__ == "__main__":
    main()

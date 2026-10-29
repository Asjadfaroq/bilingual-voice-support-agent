import asyncio

from psycopg import AsyncConnection

from voice_agent.config import get_settings
from voice_agent.knowledge.ingest import (
    build_sentence_embedder,
    create_knowledge_table,
    load_knowledge_chunks,
    replace_knowledge_chunks,
)


async def seed_knowledge_base() -> int:
    settings = get_settings()
    chunks = load_knowledge_chunks(settings.kb_directory, settings.kb_chunk_max_characters)
    embed = build_sentence_embedder(settings.embedding_model)
    async with await AsyncConnection.connect(settings.database_url) as connection:
        await create_knowledge_table(connection, settings.embedding_dimensions)
        return await replace_knowledge_chunks(connection, chunks, embed)


if __name__ == "__main__":
    stored_count = asyncio.run(seed_knowledge_base())
    print(f"Stored {stored_count} knowledge base chunks")

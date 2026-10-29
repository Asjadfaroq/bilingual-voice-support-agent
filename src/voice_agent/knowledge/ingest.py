import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import yaml
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection

TextEmbedder = Callable[[list[str]], list[list[float]]]

# multilingual-e5 is trained with these prefixes; leaving them out measurably hurts retrieval.
PASSAGE_PREFIX = "passage: "

FRONTMATTER_PATTERN = re.compile(r"\A---\n(?P<frontmatter>.*?)\n---\n", re.DOTALL)
SECTION_HEADING_PATTERN = re.compile(r"^## +(?P<heading>.+)$", re.MULTILINE)
INTRO_SECTION_SLUG = "intro"


@dataclass(frozen=True)
class KnowledgeDocument:
    slug: str
    title: str
    body: str


@dataclass(frozen=True)
class KnowledgeChunk:
    source_id: str
    title: str
    heading: str
    content: str


def read_knowledge_document(path: Path) -> KnowledgeDocument:
    text = path.read_text(encoding="utf-8")
    frontmatter_match = FRONTMATTER_PATTERN.match(text)
    if frontmatter_match is None:
        raise ValueError(f"Knowledge base document {path} has no frontmatter block")
    frontmatter = yaml.safe_load(frontmatter_match["frontmatter"])
    if "title" not in frontmatter:
        raise ValueError(f"Knowledge base document {path} has no title in its frontmatter")
    body = text[frontmatter_match.end() :]
    return KnowledgeDocument(slug=path.stem, title=frontmatter["title"], body=body)


def slugify(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def split_into_sections(body: str) -> list[tuple[str | None, str]]:
    headings = list(SECTION_HEADING_PATTERN.finditer(body))
    intro_end = headings[0].start() if headings else len(body)
    intro = re.sub(r"^# .+$", "", body[:intro_end], flags=re.MULTILINE).strip()
    sections: list[tuple[str | None, str]] = [(None, intro)] if intro else []
    for index, heading_match in enumerate(headings):
        section_end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        sections.append((heading_match["heading"].strip(), body[heading_match.end() : section_end]))
    return [(heading, text.strip()) for heading, text in sections if text.strip()]


def split_into_passages(text: str, max_characters: int) -> list[str]:
    passages: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and len(candidate) > max_characters:
            passages.append(current)
            current = paragraph
        else:
            current = candidate
    passages.append(current)
    return passages


def chunk_document(document: KnowledgeDocument, max_characters: int) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for heading, section_text in split_into_sections(document.body):
        section_slug = slugify(heading) if heading else INTRO_SECTION_SLUG
        passages = split_into_passages(section_text, max_characters)
        for passage_number, passage in enumerate(passages, start=1):
            suffix = f"-{passage_number}" if len(passages) > 1 else ""
            chunks.append(
                KnowledgeChunk(
                    source_id=f"{document.slug}#{section_slug}{suffix}",
                    title=document.title,
                    heading=heading or document.title,
                    content=passage,
                )
            )
    return chunks


def load_knowledge_chunks(kb_directory: Path, max_characters: int) -> list[KnowledgeChunk]:
    paths = sorted(kb_directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown documents found in {kb_directory}")
    return [
        chunk
        for path in paths
        for chunk in chunk_document(read_knowledge_document(path), max_characters)
    ]


def sentence_transformers_installed() -> bool:
    # pyproject leaves sentence-transformers and torch out on Intel Macs, which run ONNX instead.
    return find_spec("sentence_transformers") is not None


def build_sentence_embedder(model_name: str) -> TextEmbedder:
    if not sentence_transformers_installed():
        from voice_agent.knowledge.onnx_models import build_onnx_sentence_embedder

        return build_onnx_sentence_embedder(model_name)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, normalize_embeddings=True).tolist()


def text_to_embed(chunk: KnowledgeChunk) -> str:
    return f"{PASSAGE_PREFIX}{chunk.title}. {chunk.heading}\n{chunk.content}"


async def create_knowledge_table(connection: AsyncConnection, embedding_dimensions: int) -> None:
    await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await register_vector_async(connection)
    await connection.execute(
        "CREATE TABLE IF NOT EXISTS knowledge_chunks ("
        " source_id text PRIMARY KEY,"
        " title text NOT NULL,"
        " heading text NOT NULL,"
        " content text NOT NULL,"
        f" embedding vector({int(embedding_dimensions)}) NOT NULL)"
    )


async def replace_knowledge_chunks(
    connection: AsyncConnection, chunks: list[KnowledgeChunk], embed: TextEmbedder
) -> int:
    embeddings = await asyncio.to_thread(embed, [text_to_embed(chunk) for chunk in chunks])
    await connection.execute("DELETE FROM knowledge_chunks")
    async with connection.cursor() as cursor:
        await cursor.executemany(
            "INSERT INTO knowledge_chunks (source_id, title, heading, content, embedding)"
            " VALUES (%s, %s, %s, %s, %s::vector)",
            [
                (chunk.source_id, chunk.title, chunk.heading, chunk.content, embedding)
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )
    return len(chunks)

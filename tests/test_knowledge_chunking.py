from pathlib import Path

import pytest

from voice_agent.knowledge.ingest import (
    KnowledgeDocument,
    chunk_document,
    load_knowledge_chunks,
    read_knowledge_document,
    text_to_embed,
)

KB_DIRECTORY = Path(__file__).parent.parent / "data" / "kb"

SAMPLE_BODY = """# Ledgerly Product FAQ

Ledgerly is an invoicing product.

## What plans does Ledgerly offer?

Starter costs 49 SAR per month.

## Is there a free trial?

Every plan starts with a 14-day free trial.
"""


def write_document(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_title_comes_from_frontmatter_and_slug_from_file_name(tmp_path: Path) -> None:
    path = write_document(
        tmp_path, "product-faq.md", f"---\ntitle: Product FAQ\n---\n{SAMPLE_BODY}"
    )

    document = read_knowledge_document(path)

    assert document.title == "Product FAQ"
    assert document.slug == "product-faq"
    assert document.body.startswith("# Ledgerly Product FAQ")


def test_document_without_frontmatter_is_rejected_with_its_path(tmp_path: Path) -> None:
    path = write_document(tmp_path, "broken.md", SAMPLE_BODY)

    with pytest.raises(ValueError, match=r"broken\.md has no frontmatter"):
        read_knowledge_document(path)


def test_each_second_level_heading_becomes_its_own_chunk() -> None:
    document = KnowledgeDocument(slug="faq", title="Product FAQ", body=SAMPLE_BODY)

    chunks = chunk_document(document, max_characters=1000)

    assert [chunk.source_id for chunk in chunks] == [
        "faq#intro",
        "faq#what-plans-does-ledgerly-offer",
        "faq#is-there-a-free-trial",
    ]
    assert chunks[1].heading == "What plans does Ledgerly offer?"
    assert chunks[1].content == "Starter costs 49 SAR per month."


def test_intro_text_is_kept_under_the_document_title_without_the_top_heading() -> None:
    document = KnowledgeDocument(slug="faq", title="Product FAQ", body=SAMPLE_BODY)

    intro = chunk_document(document, max_characters=1000)[0]

    assert intro.heading == "Product FAQ"
    assert intro.content == "Ledgerly is an invoicing product."


def test_long_section_is_split_on_paragraph_boundaries_within_the_limit() -> None:
    paragraphs = [f"Paragraph {number}" + " word" * 20 for number in range(4)]
    body = "## Limits\n\n" + "\n\n".join(paragraphs)
    document = KnowledgeDocument(slug="policy", title="Policy", body=body)

    chunks = chunk_document(document, max_characters=250)

    assert [chunk.source_id for chunk in chunks] == [
        "policy#limits-1",
        "policy#limits-2",
    ]
    assert all(len(chunk.content) <= 250 for chunk in chunks)
    assert "\n\n".join(chunk.content for chunk in chunks) == "\n\n".join(paragraphs)


def test_passage_text_carries_e5_prefix_title_and_heading() -> None:
    document = KnowledgeDocument(slug="faq", title="Product FAQ", body=SAMPLE_BODY)
    plans_chunk = chunk_document(document, max_characters=1000)[1]

    assert text_to_embed(plans_chunk) == (
        "passage: Product FAQ. What plans does Ledgerly offer?\nStarter costs 49 SAR per month."
    )


def test_every_bundled_kb_document_produces_uniquely_identified_chunks() -> None:
    chunks = load_knowledge_chunks(KB_DIRECTORY, max_characters=1200)

    source_ids = [chunk.source_id for chunk in chunks]
    assert len(source_ids) == len(set(source_ids))
    assert {source_id.split("#")[0] for source_id in source_ids} == {
        path.stem for path in KB_DIRECTORY.glob("*.md")
    }


def test_empty_kb_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No markdown documents"):
        load_knowledge_chunks(tmp_path, max_characters=1200)

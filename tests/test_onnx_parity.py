import numpy as np
import pytest

from voice_agent.config import Settings
from voice_agent.knowledge.ingest import sentence_transformers_installed
from voice_agent.knowledge.onnx_models import (
    build_onnx_passage_scorer,
    build_onnx_sentence_embedder,
)

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        not sentence_transformers_installed(),
        reason="sentence-transformers is not installed on this platform (Intel Mac)",
    ),
]

MIN_COSINE_SIMILARITY = 0.999

PARITY_SENTENCES = [
    "query: How do I reset my password?",
    "query: Can I export my invoices as a CSV file?",
    "passage: Support is open Sunday to Thursday, from nine in the morning to five.",
    "passage: Expense claims above five hundred riyals need a manager's approval.",
    "passage: P1 tickets get a first response within one hour.",
    "query: كيف أعيد تعيين كلمة المرور؟",
    "query: هل يمكنني تصدير فواتيري؟",
    "passage: يعمل الدعم من الأحد إلى الخميس من التاسعة صباحا حتى الخامسة.",
    "passage: تحتاج المطالبات التي تزيد على خمسمائة ريال إلى موافقة المدير.",
    "passage: نعم",
]

RERANK_QUESTION = "How fast do you answer urgent tickets?"
RERANK_PASSAGES = [
    "P1 tickets get a first response within one hour.",
    "Support is open Sunday to Thursday, from nine in the morning to five.",
    "يتم الرد على التذاكر العاجلة خلال ساعة واحدة.",
    "Expense claims above five hundred riyals need a manager's approval.",
    "You can export invoices as PDF or CSV from the billing page.",
]


def test_onnx_embeddings_match_sentence_transformers(test_settings: Settings) -> None:
    from sentence_transformers import SentenceTransformer

    model_name = test_settings.embedding_model
    reference = SentenceTransformer(model_name).encode(PARITY_SENTENCES, normalize_embeddings=True)
    onnx_embeddings = np.array(build_onnx_sentence_embedder(model_name)(PARITY_SENTENCES))

    cosine_similarities = (reference * onnx_embeddings).sum(axis=1)

    assert cosine_similarities.min() > MIN_COSINE_SIMILARITY, cosine_similarities


def test_onnx_reranker_orders_passages_like_the_cross_encoder(
    test_settings: Settings,
) -> None:
    from sentence_transformers import CrossEncoder

    model_name = test_settings.kb_rerank_model
    reference_scores = CrossEncoder(model_name).predict(
        [(RERANK_QUESTION, passage) for passage in RERANK_PASSAGES]
    )
    onnx_scores = build_onnx_passage_scorer(model_name)(RERANK_QUESTION, RERANK_PASSAGES)

    assert np.argsort(onnx_scores).tolist() == np.argsort(reference_scores).tolist()

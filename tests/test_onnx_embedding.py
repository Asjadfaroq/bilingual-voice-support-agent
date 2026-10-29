import numpy as np

from voice_agent.knowledge.onnx_models import mean_pool_and_normalize


def test_padding_tokens_do_not_change_the_sentence_embedding() -> None:
    tokens = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    padded_tokens = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, -100.0]]])

    embedding = mean_pool_and_normalize(tokens, np.array([[1, 1]]))
    padded_embedding = mean_pool_and_normalize(padded_tokens, np.array([[1, 1, 0]]))

    np.testing.assert_allclose(padded_embedding, embedding)


def test_sentence_embeddings_have_unit_length() -> None:
    tokens = np.array([[[3.0, 4.0]], [[-6.0, 8.0]]])

    embeddings = mean_pool_and_normalize(tokens, np.array([[1], [1]]))

    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), [1.0, 1.0])

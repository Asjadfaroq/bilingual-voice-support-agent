from dataclasses import dataclass

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from voice_agent.knowledge.ingest import TextEmbedder
from voice_agent.knowledge.search import PassageScorer

ONNX_MODEL_FILE = "onnx/model.onnx"
TOKENIZER_FILE = "tokenizer.json"
# Both models are XLM-R based: they compute position ids from this pad token, and their
# sentence-transformers configs cut input at 512 tokens.
PAD_TOKEN = "<pad>"
MAX_SEQUENCE_TOKENS = 512
# The same floor sentence-transformers uses, so an empty mask divides by a tiny number, not zero.
MIN_TOKEN_COUNT = 1e-9


@dataclass(frozen=True)
class OnnxModel:
    tokenizer: Tokenizer
    session: onnxruntime.InferenceSession


def load_onnx_model(model_name: str) -> OnnxModel:
    tokenizer = Tokenizer.from_file(hf_hub_download(model_name, TOKENIZER_FILE))
    tokenizer.enable_truncation(MAX_SEQUENCE_TOKENS)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id(PAD_TOKEN), pad_token=PAD_TOKEN)
    session = onnxruntime.InferenceSession(hf_hub_download(model_name, ONNX_MODEL_FILE))
    return OnnxModel(tokenizer, session)


def run_onnx_model(
    model: OnnxModel, inputs: list[str] | list[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray]:
    encodings = model.tokenizer.encode_batch(inputs)
    attention_mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
    model_inputs = {
        "input_ids": np.array([encoding.ids for encoding in encodings], dtype=np.int64),
        "attention_mask": attention_mask,
        "token_type_ids": np.array([encoding.type_ids for encoding in encodings], dtype=np.int64),
    }
    expected_names = {model_input.name for model_input in model.session.get_inputs()}
    [output] = model.session.run(
        None, {name: value for name, value in model_inputs.items() if name in expected_names}
    )
    return output, attention_mask


def mean_pool_and_normalize(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., np.newaxis].astype(token_embeddings.dtype)
    pooled = (token_embeddings * mask).sum(axis=1) / np.clip(
        mask.sum(axis=1), MIN_TOKEN_COUNT, None
    )
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)


def build_onnx_sentence_embedder(model_name: str) -> TextEmbedder:
    model = load_onnx_model(model_name)
    return lambda texts: mean_pool_and_normalize(*run_onnx_model(model, texts)).tolist()


def build_onnx_passage_scorer(model_name: str) -> PassageScorer:
    model = load_onnx_model(model_name)
    return lambda question, passages: run_onnx_model(
        model, [(question, passage) for passage in passages]
    )[0][:, 0].tolist()

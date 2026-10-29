# Decisions

- STT is ElevenLabs Scribe v2 realtime (`scribe_v2_realtime`) instead of Deepgram, which is not available; `livekit-plugins-elevenlabs` 1.8.10 streams it over a websocket and supports `secondary_languages` and per-transcript language detection, which suits Arabic/English switching.
- TTS is ElevenLabs `eleven_flash_v2_5` (lowest-latency multilingual model) with the premade voice "Sarah" (`EXAVITQu4vr4xnSDxMaL`), which ElevenLabs verifies for Arabic and English.
- HubSpot uses a service key on the EU host `api-eu1.hubapi.com`; creating a call engagement works with only the contacts and tickets scopes (tested 2026-10-29: `POST /crm/v3/objects/calls` returned 201, then deleted), so calls are logged as real call objects, not notes.
- `torch` comes from the CPU wheel index: the embedding model runs on CPU and the CUDA build grew the venv from 1.5 GB to 5.8 GB.
- `kb_min_similarity` (0.72) is a noise floor, not the "found / not found" decision: measured e5 scores overlap (Arabic in-domain questions from 0.756, English off-topic up to 0.803), so the LLM judges whether the returned passages answer the question.
- No ANN index on `knowledge_chunks`: the KB is 19 chunks, and an exact scan is both faster and exact.
- `search.py` imports the `TextEmbedder` type from `ingest.py` rather than adding an embeddings module that is not in the spec tree; ingest owns the stored format that search reads.
- `scripts/seed.py` seeds only the knowledge base until HubSpot seeding is added in step 5; the Langfuse compose profile is added with tracing in step 6.
- Turn detection uses `livekit.agents.inference.TurnDetector` (audio model, lists Arabic) instead of the deprecated `livekit-plugins-turn-detector` multilingual text model, whose language list does not include Arabic; `v1` runs on LiveKit Inference and falls back to the local `v1-mini` at the free-plan limit.
- Barge-in uses `interruption_mode="vad"` by default: local, free and identical in dev and production; `adaptive` (LiveKit Inference, ignores backchannels like "uh-huh") is one config change away.
- Claude Haiku 5.5 runs with thinking disabled and effort `low`; the LiveKit Anthropic plugin has no option for these, so `ReasoningControlledLLM` attaches them to every request.
- The worker runs with `python -m livekit.agents start ... --dev`, because the built-in `cli.run_app` CLI is deprecated in livekit-agents 1.8.
- `search_knowledge_base` asks the LLM for the question in English: the KB is English, and e5 scores English queries higher than the same question in Arabic.
- Turn latency is read from LiveKit's per-message metrics (`e2e_latency`, `llm_node_ttft`, `tts_node_ttfb`, `end_of_turn_delay`, `transcription_delay`); on knowledge turns the first audio is the agent's short acknowledgement before the tool call.

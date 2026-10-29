# bilingual-voice-support-agent

## Latency

Measured on 2026-10-30 on an Intel MacBook Pro (MacBookPro16,1, Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz, 32 GB, macOS 26.5.2), with the agent worker running on the Mac and LiveKit Cloud in between. Earlier timings in `docs/DECISIONS.md` were measured on Windows/WSL and are not comparable.

Shipped configuration (knowledge base searched before the model, 0.3 s silence wait), 20 distinct questions per language through a real LiveKit room, p50 / p90 in seconds:

| Stage | English | Arabic |
|---|---|---|
| End of caller speech → first agent audio (agent-measured) | 2.31 / 5.52 | 2.92 / 6.29 |
| End-of-turn delay (includes STT final) | 0.84 / 3.42 | 0.74 / 2.33 |
| STT final | 0.84 / 3.42 | 0.74 / 2.33 |
| LLM time to first token (includes knowledge search) | 0.92 / 1.23 | 1.17 / 1.39 |
| Reply filter hold | 0.14 / 0.19 | 0.20 / 0.52 |
| TTS first byte | 0.24 / 0.37 | 0.37 / 0.59 |
| Heard by the caller (adds both network legs) | 3.56 / 6.87 | 4.42 / 7.71 |
| Turns answered | 19 of 20 | 16 of 20 |

STT final is part of the end-of-turn delay. At p50, end of turn, LLM, filter and TTS add up to 2.14 s in English and 2.48 s in Arabic against the 2.31 s and 2.92 s measured end to end. The p50 is well above the 1 s budget, so the agent is not sub-second. Unanswered turns were lost while ElevenLabs' realtime STT reconnected after the server closed the socket, and those reconnects also make the p90 tails.

Compared configurations (p50 / p90 in seconds, heard by the caller):

| Configuration | First audio, en | First audio, ar | Answer, en | Answer, ar | Cut-offs |
|---|---|---|---|---|---|
| Search every turn, 0.55 s silence | 3.91 / 5.71 | 4.53 / 7.09 | 3.91 / 5.71 | 4.53 / 7.09 | 0 |
| Search every turn, 0.3 s silence (shipped) | 3.56 / 6.87 | 4.42 / 7.71 | 3.56 / 6.87 | 4.42 / 7.71 | 0 |
| Search tool, 0.3 s silence | 3.51 / 5.51 | 4.02 / 7.34 | 5.50 / 7.28 | 7.64 / 9.27 | 1 |

Full per-turn results: `evals/results/latency.json`.

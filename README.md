# bilingual-voice-support-agent

## Latency

Measured on an Intel MacBook Pro (MacBookPro16,1, Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz, 32 GB, macOS 26.5.2), with the agent worker running on the Mac and LiveKit Cloud in between (TCP round trip to LiveKit 52-58 ms during the shipped run). Earlier timings in `docs/DECISIONS.md` were measured on Windows/WSL and are not comparable.

Shipped configuration (knowledge base searched before the model, 0.3 s silence wait), 20 distinct questions per language through a real LiveKit room, p50 / p90 in seconds:

| Stage | English | Arabic |
|---|---|---|
| End of caller speech → first agent audio (agent-measured) | 2.30 / 4.48 | 2.50 / 4.08 |
| End-of-turn delay (includes STT final) | 0.76 / 3.00 | 0.71 / 1.73 |
| STT final | 0.76 / 2.12 | 0.71 / 1.73 |
| LLM time to first token (includes knowledge search) | 0.96 / 1.66 | 1.04 / 1.58 |
| Reply filter hold | 0.15 / 0.20 | 0.21 / 0.38 |
| TTS first byte | 0.26 / 0.47 | 0.23 / 0.33 |
| Heard by the caller | 3.70 / 5.97 | 3.53 / 5.40 |
| … of which caller → agent transport and VAD | 0.53 / 0.63 | 0.62 / 0.76 |
| … of which agent → caller transport | 0.57 / 0.99 | 0.56 / 0.88 |
| Turns answered | 20 of 20 | 20 of 20 |

STT final is part of the end-of-turn delay. The p50 is well above the 1 s budget, so the agent is not sub-second. What the caller hears is about 1.2 s later than the agent's own measurement: about 0.5-0.6 s each way is LiveKit/WebRTC transport and jitter buffering, which two plain LiveKit participants on this Mac measured too (0.40 s one way with continuous audio, 0.63 s after silence), plus the VAD's end-of-speech lag on the way in. The p90 tails come from the turn detector waiting up to its 3 s maximum on turns it judged unfinished and from slow model starts.

Before and after the STT fixes (shipped configuration, p50 / p90 in seconds):

| Run | Answered, en | Answered, ar | Heard by caller, en | Heard by caller, ar | Agent-measured, en | Agent-measured, ar |
|---|---|---|---|---|---|---|
| Before | 19 of 20 | 16 of 20 | 3.56 / 6.87 | 4.42 / 7.71 | 2.31 / 5.52 | 2.92 / 6.29 |
| 30 s liveness timeout, speech-only replay | 19 of 20 | 17 of 20 | 3.36 / 3.92 | 3.90 / 4.99 | 2.33 / 2.71 | 2.69 / 3.65 |
| … and the speech gate (shipped) | 20 of 20 | 20 of 20 | 3.70 / 5.97 | 3.53 / 5.40 | 2.30 / 4.48 | 2.50 / 4.08 |

The shipped run's p90 is higher than the middle run's because it answers, and so counts, turns the earlier runs lost: two of them are utterances with a mid-sentence pause, on which the turn detector waits up to 3 s, and one English turn took 11.9 s (a 7.7 s end-of-turn wait and a 3.8 s model start).

Knowledge search designs and silence waits compared (before the STT fixes; p50 / p90 in seconds, heard by the caller):

| Configuration | First audio, en | First audio, ar | Answer, en | Answer, ar | Cut-offs |
|---|---|---|---|---|---|
| Search every turn, 0.55 s silence | 3.91 / 5.71 | 4.53 / 7.09 | 3.91 / 5.71 | 4.53 / 7.09 | 0 |
| Search every turn, 0.3 s silence | 3.56 / 6.87 | 4.42 / 7.71 | 3.56 / 6.87 | 4.42 / 7.71 | 0 |
| Search tool, 0.3 s silence | 3.51 / 5.51 | 4.02 / 7.34 | 5.50 / 7.28 | 7.64 / 9.27 | 1 |

Full per-turn results: `evals/results/latency.json`.

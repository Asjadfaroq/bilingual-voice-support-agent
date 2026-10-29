from livekit.agents import AgentServer, AgentSession, JobContext, JobProcess
from psycopg_pool import AsyncConnectionPool

from voice_agent.config import get_settings
from voice_agent.conversation.support_agent import SupportAgent
from voice_agent.conversation.tools import CallResources
from voice_agent.knowledge.ingest import build_sentence_embedder
from voice_agent.knowledge.search import build_cross_encoder_reranker
from voice_agent.metrics.turn_latency import log_latency_for_each_turn
from voice_agent.pipeline.components import (
    build_llm,
    build_stt,
    build_tts,
    build_turn_handling,
    build_vad,
)

settings = get_settings()


def load_models_once_per_process(process: JobProcess) -> None:
    # Loading the embedding model takes seconds, so it happens before a call is assigned.
    process.userdata["embed"] = build_sentence_embedder(settings.embedding_model)
    process.userdata["reranker"] = (
        build_cross_encoder_reranker(settings.kb_rerank_model, settings.kb_rerank_candidate_count)
        if settings.kb_rerank_enabled
        else None
    )
    process.userdata["vad"] = build_vad(settings)


server = AgentServer(
    setup_fnc=load_models_once_per_process,
    initialize_process_timeout=settings.worker_initialize_timeout_seconds,
    num_idle_processes=settings.worker_idle_processes,
    ws_url=settings.livekit_url,
    api_key=settings.livekit_api_key.get_secret_value(),
    api_secret=settings.livekit_api_secret.get_secret_value(),
)


@server.rtc_session()
async def run_support_call(ctx: JobContext) -> None:
    # Not opened with wait=True: if the database is down the call still starts and the
    # knowledge tool apologises instead of the caller hearing nothing.
    knowledge_pool = AsyncConnectionPool(
        settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_connection_timeout_seconds,
        open=False,
    )
    await knowledge_pool.open()

    async def close_knowledge_pool() -> None:
        await knowledge_pool.close()

    ctx.add_shutdown_callback(close_knowledge_pool)

    session = AgentSession[CallResources](
        stt=build_stt(settings),
        llm=build_llm(settings),
        tts=build_tts(settings),
        vad=ctx.proc.userdata["vad"],
        turn_handling=build_turn_handling(settings),
        userdata=CallResources(
            knowledge_pool=knowledge_pool,
            embed=ctx.proc.userdata["embed"],
            reranker=ctx.proc.userdata["reranker"],
            kb_top_k=settings.kb_top_k,
            kb_min_similarity=settings.kb_min_similarity,
        ),
    )
    log_latency_for_each_turn(session, settings.latency_budget_seconds)
    await session.start(agent=SupportAgent(), room=ctx.room)

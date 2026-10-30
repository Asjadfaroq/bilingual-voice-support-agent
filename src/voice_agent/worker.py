import logging
from datetime import UTC, datetime

import httpx
from livekit.agents import AgentServer, AgentSession, JobContext, JobProcess
from psycopg_pool import AsyncConnectionPool

from voice_agent.config import Settings, get_settings
from voice_agent.conversation.call_state import (
    CallResources,
    CallState,
    caller_phone_from_metadata,
)
from voice_agent.conversation.support_agent import SupportAgent
from voice_agent.conversation.tools import identify_caller_by_line
from voice_agent.crm.call_log import CallRecord, build_call_summariser, log_call
from voice_agent.crm.hubspot_client import HubSpotClient, HubSpotError
from voice_agent.knowledge.ingest import build_sentence_embedder
from voice_agent.knowledge.search import build_cross_encoder_reranker
from voice_agent.metrics.tracing import CallTraceContext, start_call_tracing
from voice_agent.metrics.turn_latency import log_latency_for_each_turn, reply_filter_hold_reporter
from voice_agent.pipeline.components import (
    build_llm,
    build_stt,
    build_tts,
    build_turn_handling,
    build_vad,
)
from voice_agent.pipeline.language import LanguageSettings
from voice_agent.privacy.pii import PiiMaskingFilter

logger = logging.getLogger(__name__)
settings = get_settings()


def build_hubspot_client(settings: Settings) -> HubSpotClient:
    http = httpx.AsyncClient(
        base_url=settings.hubspot_base_url,
        headers={"Authorization": f"Bearer {settings.hubspot_access_token.get_secret_value()}"},
        timeout=settings.hubspot_timeout_seconds,
    )
    return HubSpotClient(
        http, settings.hubspot_max_attempts, settings.hubspot_retry_backoff_seconds
    )


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


@server.rtc_session(agent_name=settings.agent_name)
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
    for handler in logging.getLogger().handlers:
        handler.addFilter(PiiMaskingFilter())
    crm = build_hubspot_client(settings)
    call_state = CallState(
        caller_phone=caller_phone_from_metadata(
            ctx.job.room.metadata, settings.caller_phone_metadata_key
        )
    )

    reply_tts = build_tts(settings)
    agent = SupportAgent(
        LanguageSettings(
            default_voice_id=settings.tts_voice_id,
            voice_id_by_language=settings.tts_voice_id_by_language,
            script_dominance_threshold=settings.script_dominance_threshold,
            reply_language_min_letters=settings.reply_language_min_letters,
        ),
        reply_tts=reply_tts,
        call=call_state,
        kb_search_mode=settings.kb_search_mode,
        report_reply_filter_hold=reply_filter_hold_reporter(settings.turn_latency_topic),
    )
    trace_provider = start_call_tracing(
        settings,
        CallTraceContext(
            call_id=ctx.job.room.name,
            caller_key=lambda: (
                call_state.caller_phone or (call_state.contact.id if call_state.contact else None)
            ),
            languages=lambda: list(call_state.languages),
        ),
    )
    session = AgentSession[CallResources](
        stt=build_stt(settings),
        llm=build_llm(settings),
        tts=reply_tts,
        vad=ctx.proc.userdata["vad"],
        turn_handling=build_turn_handling(settings),
        userdata=CallResources(
            knowledge_pool=knowledge_pool,
            embed=ctx.proc.userdata["embed"],
            reranker=ctx.proc.userdata["reranker"],
            kb_top_k=settings.kb_top_k,
            kb_min_similarity=settings.kb_min_similarity,
            crm=crm,
            ticket_pipeline_id=settings.hubspot_ticket_pipeline_id,
            new_ticket_stage_id=settings.hubspot_new_ticket_stage_id,
            ticket_status_limit=settings.ticket_status_limit,
            default_country_calling_code=settings.default_country_calling_code,
            reply_language_min_letters=settings.reply_language_min_letters,
            call=call_state,
        ),
    )
    log_latency_for_each_turn(session, settings.latency_budget_seconds, settings.turn_latency_topic)
    started_at = datetime.now(UTC)

    # One callback, because LiveKit runs shutdown callbacks concurrently and the call must be
    # logged before the HubSpot client closes.
    async def log_call_and_close() -> None:
        record = CallRecord(
            started_at=started_at,
            ended_at=datetime.now(UTC),
            languages=call_state.languages,
            contact_id=call_state.contact.id if call_state.contact else None,
            ticket_ids=call_state.ticket_ids,
        )
        try:
            if settings.call_logging_enabled:
                await log_call(crm, record, session.history, build_call_summariser(settings))
        except HubSpotError as error:
            logger.error("End-of-call logging failed: %s", error)
        await crm.aclose()
        await knowledge_pool.close()
        if trace_provider is not None:
            trace_provider.shutdown()

    ctx.add_shutdown_callback(log_call_and_close)
    await session.start(agent=agent, room=ctx.room)
    # After start, so the greeting plays while HubSpot answers.
    await identify_caller_by_line(session.userdata)

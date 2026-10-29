from livekit.agents import Agent

from voice_agent.conversation.instructions import GREETING, SUPPORT_INSTRUCTIONS
from voice_agent.conversation.tools import search_knowledge_base


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SUPPORT_INSTRUCTIONS, tools=[search_knowledge_base])

    async def on_enter(self) -> None:
        self.session.say(GREETING)

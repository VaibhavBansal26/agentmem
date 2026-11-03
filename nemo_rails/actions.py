from nemoguardrails.actions import action

from agentmem.guardrails import TopicalRail

_rail = TopicalRail()


@action(name="check_topic")
async def check_topic(text: str = "") -> bool:
    return _rail.check_input(text).safe

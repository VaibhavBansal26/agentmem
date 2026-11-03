"""Runs the NeMo Guardrails topical input rail (no LLM). Skips if nemoguardrails is not installed."""
import asyncio

try:
    from nemoguardrails import LLMRails, RailsConfig
except ImportError:
    LLMRails = None


def test_nemo_topical_input_rail():
    if LLMRails is None:
        print("nemoguardrails not installed: skipped")
        return
    rails = LLMRails(RailsConfig.from_path("nemo_rails"))

    async def ask(q):
        res = await rails.generate_async(messages=[{"role": "user", "content": q}], options={"rails": ["input"]})
        return str(getattr(res, "response", res))

    assert "can\'t speculate" in asyncio.run(ask("Did she intend to leak the report?"))
    assert "can\'t speculate" not in asyncio.run(ask("What happened on web-12?"))


if __name__ == "__main__":
    test_nemo_topical_input_rail()
    print("nemo ok")

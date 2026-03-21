from __future__ import annotations

from collections.abc import Sequence

import httpx
from kosong.chat_provider import ChatProviderError, TokenUsage
from kosong.chat_provider.openai_common import convert_error
from kosong.message import Message

from kimi_cli.llm import LLM
from kimi_cli.soul.compaction import CompactionResult, SimpleCompaction
from kimi_cli.wire.types import ContentPart, TextPart

MORPH_COMPACTION_QUERY = (
    "Keep only the context that is still necessary to continue this coding task accurately. "
    "Preserve user goals, decisions, concrete file paths, commands, errors, active plans, "
    "code changes, and unresolved blockers. Remove redundant chatter and stale details."
)


class MorphCompaction(SimpleCompaction):
    def __init__(self, max_preserved_messages: int = 2, *, timeout_s: float = 60.0) -> None:
        super().__init__(max_preserved_messages=max_preserved_messages)
        self.timeout_s = timeout_s

    async def compact(
        self, messages: Sequence[Message], llm: LLM, *, custom_instruction: str = ""
    ) -> CompactionResult:
        split = self._split_messages(messages)
        if not split.to_compact:
            return CompactionResult(messages=split.to_preserve, usage=None)

        provider = llm.provider_config
        if provider is None:
            raise ChatProviderError("Morph compaction requires provider configuration.")

        base_url = getattr(llm.chat_provider, "_base_url", None) or provider.base_url
        if not base_url:
            raise ChatProviderError("Morph compaction requires a provider base_url.")

        api_key = (
            getattr(llm.chat_provider, "_api_key", None) or provider.api_key.get_secret_value()
        )
        if not api_key:
            raise ChatProviderError("Morph compaction requires an API key.")

        model_name = llm.model_config.model if llm.model_config is not None else llm.model_name
        if not model_name:
            raise ChatProviderError("Morph compaction requires a configured model name.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if provider.custom_headers:
            headers.update(provider.custom_headers)

        payload: dict[str, object] = {
            "model": model_name,
            "messages": [self._message_to_payload(message) for message in split.to_compact],
            "query": self._build_query(custom_instruction),
            "compression_ratio": 0.5,
            "compress_system_messages": False,
        }

        url = base_url.rstrip("/") + "/compact"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise convert_error(error) from error

        data = response.json()
        compacted_messages = self._parse_messages(data.get("messages"))
        compacted_messages.extend(split.to_preserve)
        return CompactionResult(
            messages=compacted_messages,
            usage=self._parse_usage(data.get("usage")),
        )

    @staticmethod
    def _message_to_payload(message: Message) -> dict[str, str]:
        text = "\n".join(part.text for part in message.content if isinstance(part, TextPart))
        return {"role": message.role, "content": text}

    @staticmethod
    def _build_query(custom_instruction: str) -> str:
        if not custom_instruction:
            return MORPH_COMPACTION_QUERY
        return (
            MORPH_COMPACTION_QUERY
            + " Prioritize this user instruction during compaction: "
            + custom_instruction
        )

    def _parse_messages(self, raw_messages: object) -> list[Message]:
        if not isinstance(raw_messages, list):
            raise ChatProviderError("Morph compaction returned an invalid messages payload.")

        messages: list[Message] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                raise ChatProviderError("Morph compaction returned a non-object message.")
            role = item.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ChatProviderError("Morph compaction returned a message with an invalid role.")
            content = self._parse_content(item.get("content"))
            messages.append(Message(role=role, content=content))
        return messages

    def _parse_content(self, raw_content: object) -> list[ContentPart]:
        if isinstance(raw_content, str):
            return [TextPart(text=raw_content)]
        if not isinstance(raw_content, list):
            raise ChatProviderError("Morph compaction returned invalid message content.")

        parts: list[ContentPart] = []
        for item in raw_content:
            if isinstance(item, str):
                parts.append(TextPart(text=item))
                continue
            if not isinstance(item, dict):
                raise ChatProviderError("Morph compaction returned an invalid content part.")
            try:
                parts.append(ContentPart.model_validate(item))
            except Exception as error:
                raise ChatProviderError(
                    f"Morph compaction returned an unsupported content part: {error}"
                ) from error
        return parts

    @staticmethod
    def _parse_usage(raw_usage: object) -> TokenUsage | None:
        if raw_usage is None:
            return None
        if not isinstance(raw_usage, dict):
            raise ChatProviderError("Morph compaction returned an invalid usage payload.")
        input_tokens = raw_usage.get("input_tokens", 0)
        output_tokens = raw_usage.get("output_tokens", 0)
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            raise ChatProviderError("Morph compaction usage values must be integers.")
        return TokenUsage(input_other=input_tokens, output=output_tokens)


__all__ = ["MORPH_COMPACTION_QUERY", "MorphCompaction"]

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

import httpx
import kosong
from kosong.chat_provider import ChatProviderError, TokenUsage
from kosong.chat_provider.openai_common import convert_error
from kosong.message import Message
from kosong.tooling.empty import EmptyToolset

import kimi_cli.prompts as prompts
from kimi_cli.llm import LLM
from kimi_cli.soul.message import system
from kimi_cli.utils.logging import logger
from kimi_cli.wire.types import ContentPart, TextPart, ThinkPart


class CompactionResult(NamedTuple):
    messages: Sequence[Message]
    usage: TokenUsage | None

    @property
    def estimated_token_count(self) -> int:
        """Estimate the token count of the compacted messages.

        When LLM usage is available, ``usage.output`` gives the exact token count
        of the generated summary (the first message).  Preserved messages (all
        subsequent messages) are estimated from their text length.

        When usage is not available (no compaction LLM call was made), all
        messages are estimated from text length.

        The estimate is intentionally conservative — it will be replaced by the
        real value on the next LLM call.
        """
        if self.usage is not None and len(self.messages) > 0:
            summary_tokens = self.usage.output
            preserved_tokens = estimate_text_tokens(self.messages[1:])
            return summary_tokens + preserved_tokens

        return estimate_text_tokens(self.messages)


MORPH_COMPACTOR_MODEL = "morph-compactor"
MORPH_COMPACTION_QUERY = (
    "Keep only the context that is still necessary to continue this coding task accurately. "
    "Preserve user goals, decisions, concrete file paths, commands, errors, active plans, "
    "code changes, and unresolved blockers. Remove redundant chatter and stale details."
)


class CompactionSplit(NamedTuple):
    to_compact: Sequence[Message]
    to_preserve: Sequence[Message]


def estimate_text_tokens(messages: Sequence[Message]) -> int:
    """Estimate tokens from message text content using a character-based heuristic."""
    total_chars = 0
    for msg in messages:
        for part in msg.content:
            if isinstance(part, TextPart):
                total_chars += len(part.text)
    # ~4 chars per token for English; somewhat underestimates for CJK text,
    # but this is a temporary estimate that gets corrected on the next LLM call.
    return total_chars // 4


def should_auto_compact(
    token_count: int,
    max_context_size: int,
    *,
    trigger_ratio: float,
    reserved_context_size: int,
) -> bool:
    """Determine whether auto-compaction should be triggered.

    Returns True when either condition is met (whichever fires first):
    - Ratio-based: token_count >= max_context_size * trigger_ratio
    - Reserved-based: token_count + reserved_context_size >= max_context_size
    """
    return (
        token_count >= max_context_size * trigger_ratio
        or token_count + reserved_context_size >= max_context_size
    )


@runtime_checkable
class Compaction(Protocol):
    async def compact(
        self, messages: Sequence[Message], llm: LLM, *, custom_instruction: str = ""
    ) -> CompactionResult:
        """
        Compact a sequence of messages into a new sequence of messages.

        Args:
            messages (Sequence[Message]): The messages to compact.
            llm (LLM): The LLM to use for compaction.
            custom_instruction: Optional user instruction to guide compaction focus.

        Returns:
            CompactionResult: The compacted messages and token usage from the compaction LLM call.

        Raises:
            ChatProviderError: When the chat provider returns an error.
        """
        ...


class NativeCompactor(Protocol):
    def matches(self, llm: LLM) -> bool: ...

    async def compact(
        self,
        to_compact: Sequence[Message],
        to_preserve: Sequence[Message],
        llm: LLM,
        *,
        custom_instruction: str = "",
    ) -> CompactionResult: ...


if TYPE_CHECKING:

    def type_check(simple: SimpleCompaction):
        _: Compaction = simple


def _configured_model_name(llm: LLM) -> str:
    if llm.model_config is not None:
        return llm.model_config.model
    return llm.model_name


class MorphNativeCompactor:
    def matches(self, llm: LLM) -> bool:
        return _configured_model_name(llm).strip().lower() == MORPH_COMPACTOR_MODEL

    async def compact(
        self,
        to_compact: Sequence[Message],
        to_preserve: Sequence[Message],
        llm: LLM,
        *,
        custom_instruction: str = "",
    ) -> CompactionResult:
        provider = llm.provider_config
        if provider is None:
            raise ChatProviderError("Morph compaction requires provider configuration.")

        headers = {
            "Authorization": f"Bearer {provider.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        if provider.custom_headers:
            headers.update(provider.custom_headers)

        payload: dict[str, Any] = {
            "model": _configured_model_name(llm),
            "messages": self._messages_to_payload(to_compact),
            "query": self._build_query(custom_instruction),
            "compression_ratio": 0.5,
            "compress_system_messages": False,
        }

        endpoint = f"{provider.base_url.rstrip('/')}/compact"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise convert_error(error) from error

        compacted_messages, usage = self._parse_response(response.json())
        compacted_messages.extend(to_preserve)
        return CompactionResult(messages=compacted_messages, usage=usage)

    @staticmethod
    def _messages_to_payload(messages: Sequence[Message]) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = []
        for message in messages:
            payload.append(
                {
                    "role": message.role,
                    "content": message.extract_text(),
                }
            )
        return payload

    @staticmethod
    def _build_query(custom_instruction: str) -> str:
        query = MORPH_COMPACTION_QUERY
        if custom_instruction.strip():
            query += (
                " Prioritize this user instruction during compaction: "
                f"{custom_instruction.strip()}"
            )
        return query

    @staticmethod
    def _parse_response(payload: object) -> tuple[list[Message], TokenUsage | None]:
        if not isinstance(payload, dict):
            raise ChatProviderError("Morph compaction returned an invalid response payload.")

        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            output = payload.get("output")
            if isinstance(output, str) and output:
                return [Message(role="user", content=output)], None
            raise ChatProviderError("Morph compaction response did not include compacted messages.")

        messages: list[Message] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                raise ChatProviderError("Morph compaction returned an invalid message entry.")
            role = raw_message.get("role")
            content = raw_message.get("content")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ChatProviderError("Morph compaction returned an unsupported message role.")
            messages.append(Message.model_validate({"role": role, "content": content}))

        usage_payload = payload.get("usage")
        usage: TokenUsage | None = None
        if isinstance(usage_payload, dict):
            input_tokens = usage_payload.get("input_tokens")
            output_tokens = usage_payload.get("output_tokens")
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                usage = TokenUsage(input_other=input_tokens, output=output_tokens)

        return messages, usage


NATIVE_COMPACTORS: tuple[NativeCompactor, ...] = (MorphNativeCompactor(),)


def resolve_native_compactor(llm: LLM) -> NativeCompactor | None:
    for compactor in NATIVE_COMPACTORS:
        if compactor.matches(llm):
            return compactor
    return None


class SimpleCompaction:
    def __init__(self, max_preserved_messages: int = 2) -> None:
        self.max_preserved_messages = max_preserved_messages

    async def compact(
        self, messages: Sequence[Message], llm: LLM, *, custom_instruction: str = ""
    ) -> CompactionResult:
        split = self._split_messages(messages)
        if not split.to_compact:
            return CompactionResult(messages=split.to_preserve, usage=None)

        native_compactor = resolve_native_compactor(llm)
        if native_compactor is not None:
            return await native_compactor.compact(
                split.to_compact,
                split.to_preserve,
                llm,
                custom_instruction=custom_instruction,
            )

        compact_message, to_preserve = self.prepare(messages, custom_instruction=custom_instruction)
        if compact_message is None:
            return CompactionResult(messages=to_preserve, usage=None)

        # Call kosong.step to get the compacted context
        # TODO: set max completion tokens
        logger.debug("Compacting context...")
        result = await kosong.step(
            chat_provider=llm.chat_provider,
            system_prompt="You are a helpful assistant that compacts conversation context.",
            toolset=EmptyToolset(),
            history=[compact_message],
        )
        if result.usage:
            logger.debug(
                "Compaction used {input} input tokens and {output} output tokens",
                input=result.usage.input,
                output=result.usage.output,
            )

        content: list[ContentPart] = [
            system("Previous context has been compacted. Here is the compaction output:")
        ]
        compacted_msg = result.message

        # drop thinking parts if any
        content.extend(part for part in compacted_msg.content if not isinstance(part, ThinkPart))
        compacted_messages: list[Message] = [Message(role="user", content=content)]
        compacted_messages.extend(to_preserve)
        return CompactionResult(messages=compacted_messages, usage=result.usage)

    def _split_messages(self, messages: Sequence[Message]) -> CompactionSplit:
        if not messages or self.max_preserved_messages <= 0:
            return CompactionSplit(to_compact=[], to_preserve=messages)

        history = list(messages)
        preserve_start_index = len(history)
        n_preserved = 0
        for index in range(len(history) - 1, -1, -1):
            if history[index].role in {"user", "assistant"}:
                n_preserved += 1
                if n_preserved == self.max_preserved_messages:
                    preserve_start_index = index
                    break

        if n_preserved < self.max_preserved_messages:
            return CompactionSplit(to_compact=[], to_preserve=messages)

        to_compact = history[:preserve_start_index]
        to_preserve = history[preserve_start_index:]
        if not to_compact:
            return CompactionSplit(to_compact=[], to_preserve=to_preserve)
        return CompactionSplit(to_compact=to_compact, to_preserve=to_preserve)

    class PrepareResult(NamedTuple):
        compact_message: Message | None
        to_preserve: Sequence[Message]

    def prepare(
        self, messages: Sequence[Message], *, custom_instruction: str = ""
    ) -> PrepareResult:
        split = self._split_messages(messages)
        to_compact = split.to_compact
        to_preserve = split.to_preserve
        if not to_compact:
            return self.PrepareResult(compact_message=None, to_preserve=to_preserve)

        # Create input message for compaction
        compact_message = Message(role="user", content=[])
        for i, msg in enumerate(to_compact):
            compact_message.content.append(
                TextPart(text=f"## Message {i + 1}\nRole: {msg.role}\nContent:\n")
            )
            compact_message.content.extend(
                part for part in msg.content if isinstance(part, TextPart)
            )
        prompt_text = "\n" + prompts.COMPACT
        if custom_instruction:
            prompt_text += (
                "\n\n**User's Custom Compaction Instruction:**\n"
                "The user has specifically requested the following focus during compaction. "
                "You MUST prioritize this instruction above the default compression priorities:\n"
                f"{custom_instruction}"
            )
        compact_message.content.append(TextPart(text=prompt_text))
        return self.PrepareResult(compact_message=compact_message, to_preserve=to_preserve)

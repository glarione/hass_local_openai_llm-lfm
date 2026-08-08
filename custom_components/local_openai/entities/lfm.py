"""LFM model-specific entity support."""

from __future__ import annotations

import ast
import logging
import re
from typing import TYPE_CHECKING

from homeassistant.helpers import llm

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


_LOGGER = logging.getLogger(__name__)


def _parse_python_function_call(call_str: str, allowed_tools: set[str]) -> dict | None:
    """
    Parse a Python function call string into name and args.

    SECURITY: Only allows functions in the allowed_tools set.

    Input: 'get_candidate_status(candidate_id="12345")'
    Output: {"name": "get_candidate_status", "args": {"candidate_id": "12345"}} or None if not allowed
    """
    try:
        tree = ast.parse(call_str, mode="eval")

        if isinstance(tree.body, ast.Call):
            call = tree.body
            func_name = call.func.id if isinstance(call.func, ast.Name) else None

            # SECURITY CHECK: Validate function name against allowed tools
            if not func_name or func_name not in allowed_tools:
                _LOGGER.warning(
                    "Attempted to call unknown function: %s (allowed: %s)",
                    func_name,
                    allowed_tools,
                )
                return None

            kwargs = {}
            for keyword in call.keywords:
                if keyword.arg:
                    try:
                        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                    except (ValueError, SyntaxError):
                        _LOGGER.warning(
                            "Invalid argument value for %s.%s", func_name, keyword.arg
                        )
                        return None

            return {"name": func_name, "args": kwargs}

        _LOGGER.warning("Unexpected AST node type: %s", type(tree.body))
        return None

    except Exception as e:
        _LOGGER.error("Failed to parse LFM tool call '%s': %s", call_str, e)
        return None


def _parse_lfm_tool_calls(content: str, allowed_tools: set[str]) -> list[dict]:
    """
    Parse LFM-style tool calls from content.

    Input: "[get_candidate_status(candidate_id="12345")]"
    Output: [{"name": "get_candidate_status", "args": {"candidate_id": "12345"}}]
    """
    # Extract content between square brackets
    match = re.search(r"\[(.+)\]", content)
    if not match:
        return []

    tool_call_str = match.group(1).strip()

    # Try parsing as single call first
    parsed = _parse_python_function_call(tool_call_str, allowed_tools)
    if parsed:
        return [parsed]

    # Try parsing multiple calls
    # Split on '), ' but be careful with nested structures
    parts = re.split(r"\),\s*", tool_call_str)
    tool_calls = []

    for part in parts:
        # Restore the closing parenthesis
        call = part.strip()
        if not call.endswith(")"):
            call += ")"

        parsed = _parse_python_function_call(call, allowed_tools)
        if parsed:
            tool_calls.append(parsed)

    return tool_calls


class LfmMixin:
    """Mixin for LFM tool calling support."""

    async def _transform_lfm_stream(
        self,
        stream,
        strip_emojis: bool,
        allowed_tools: set[str] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Transform LFM streaming response, detecting and parsing tool calls.

        Args:
            stream: AsyncStream[ChatCompletionChunk]
            strip_emojis: Whether to strip emojis
            allowed_tools: Set of allowed tool names (from chat_log.llm_api.tools)

        """
        import asyncio

        import demoji

        # Import here to avoid circular imports
        from custom_components.local_openai.entity import _SUPPORTS_THINKING

        new_msg = True
        pending_think = ""
        in_think = False
        seen_visible = False
        loop = asyncio.get_running_loop()

        # LFM-specific state
        pending_lfm_tool_content = ""
        in_lfm_tool_call = False

        async for event in stream:
            chunk: dict = {}

            if not event.choices:
                continue

            choice = event.choices[0]
            delta = choice.delta

            # Handle role
            if new_msg:
                chunk["role"] = delta.role or "assistant"
                new_msg = False

            # Handle thinking content
            reasoning_content = getattr(delta, "reasoning_content", None) or getattr(
                delta, "reasoning", None
            )
            if reasoning_content:
                if _SUPPORTS_THINKING:
                    chunk["thinking_content"] = reasoning_content
                else:
                    _LOGGER.debug("LLM Thought: %s", reasoning_content)

            # Handle content
            if (content := delta.content) is not None:
                if strip_emojis:
                    content = await loop.run_in_executor(
                        None, demoji.replace, content, ""
                    )

                # Check for LFM tool call start token
                if "<|tool_call_start|>" in content:
                    in_lfm_tool_call = True
                    # Extract content after the start token
                    parts = content.split("<|tool_call_start|>")
                    if len(parts) > 1:
                        content = parts[1]
                    else:
                        content = ""

                # Check for LFM tool call end token
                if "<|tool_call_end|>" in content:
                    in_lfm_tool_call = False
                    # Extract content before the end token
                    parts = content.split("<|tool_call_end|>")
                    pending_lfm_tool_content += parts[0]

                    # Parse the tool call
                    if allowed_tools:
                        tool_calls = _parse_lfm_tool_calls(
                            pending_lfm_tool_content, allowed_tools
                        )
                        if tool_calls:
                            chunk["tool_calls"] = [
                                llm.ToolInput(
                                    id=f"call_{i}",
                                    tool_name=tc["name"],
                                    tool_args=tc["args"],
                                )
                                for i, tc in enumerate(tool_calls)
                            ]
                    pending_lfm_tool_content = ""

                    # Check if there's content after the end token
                    if len(parts) > 1:
                        content = parts[1]
                    else:
                        content = ""

                # Accumulate tool call content (don't yield it as text)
                if in_lfm_tool_call:
                    pending_lfm_tool_content += content
                    continue

                # Handle thinking tags
                if "<think>" in content:
                    in_think = True
                    content = content.replace("<think>", "")
                    pending_think = ""

                if in_think:
                    if "</think>" in content:
                        in_think = False
                        before_close, remaining = content.split("</think>", 1)
                        pending_think += before_close
                        content = remaining

                        if _SUPPORTS_THINKING:
                            if before_close:
                                chunk["thinking_content"] = before_close
                        elif pending_think.strip():
                            _LOGGER.debug("LLM Thought: %s", pending_think)
                        pending_think = ""
                    else:
                        pending_think += content
                        if _SUPPORTS_THINKING:
                            chunk["thinking_content"] = content
                        content = ""

                if not in_think and content.strip():
                    seen_visible = True

                if seen_visible:
                    chunk["content"] = content

            # Handle finish reason
            if choice.finish_reason:
                try:
                    if event.timings:
                        self.extra_state_attributes = {"timings": event.timings}
                except Exception:
                    pass

            if seen_visible or chunk.get("tool_calls") or chunk.get("role"):
                yield chunk


def get_conversation_config_schema() -> dict:
    """Return conversation config schema for LFM."""
    import voluptuous as vol

    from custom_components.local_openai.const import CONF_ENABLE_LFM_TOOL_CALLING

    return {
        vol.Required(
            CONF_ENABLE_LFM_TOOL_CALLING,
            default=False,
        ): bool,
    }


def get_ai_task_config_schema() -> dict:
    """Return AI task config schema for LFM."""
    return get_conversation_config_schema()


# Import these at the bottom to avoid circular imports
from custom_components.local_openai.ai_task import LocalAITaskEntity  # noqa: E402
from custom_components.local_openai.conversation import (
    LocalAiConversationEntity,
)


class LfmConversationEntity(LfmMixin, LocalAiConversationEntity):
    """Conversation agent with LFM tool calling support."""


class LfmAITaskEntity(LfmMixin, LocalAITaskEntity):
    """AI Task entity with LFM tool calling support."""

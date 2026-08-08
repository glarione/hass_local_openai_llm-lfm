"""LFM model-specific entity support."""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation
from homeassistant.helpers import llm
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from openai.types.chat import ChatCompletionChunk, ChatCompletionFunctionToolParam

_LOGGER = logging.getLogger(__name__)


def _parse_lfm_tool_calls(
    content: str, allowed_tools: set[str] | None = None
) -> list[dict]:
    """
    Parse LFM-style tool calls from content string using Python AST.

    Input examples:
    - "[GetLiveContext()]"
    - "[HassTurnOn(name='TV LED', area='Living Room')]"
    - "[func1(arg1='val'), func2(arg2=123)]"
    """
    if not content:
        return []

    # Extract content inside brackets if enclosed in []
    match = re.search(r"\[(.*)\]", content, re.DOTALL)
    expr_str = f"[{match.group(1).strip()}]" if match else content.strip()

    try:
        tree = ast.parse(expr_str, mode="eval")
    except Exception as err:
        _LOGGER.warning(
            "Failed to parse LFM tool call expression '%s': %s", expr_str, err
        )
        return []

    nodes: list[ast.AST] = []
    if isinstance(tree.body, ast.List):
        nodes = list(tree.body.elts)
    elif isinstance(tree.body, ast.Call):
        nodes = [tree.body]

    calls: list[dict] = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue

        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if not func_name:
            continue

        if allowed_tools is not None and func_name not in allowed_tools:
            _LOGGER.warning(
                "Attempted to call unknown function: %s (allowed: %s)",
                func_name,
                allowed_tools,
            )
            continue

        kwargs: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg:
                try:
                    kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                except Exception:
                    kwargs[keyword.arg] = str(keyword.value)

        calls.append({"name": func_name, "args": kwargs})

    return calls


def _format_lfm_tool_call_string(tool_calls: list[llm.ToolInput]) -> str:
    """Format list of ToolInput into LFM tool call text representation."""
    calls_str = []
    for call in tool_calls:
        args_formatted = []
        for k, v in call.tool_args.items():
            args_formatted.append(f"{k}={repr(v)}")
        calls_str.append(f"{call.tool_name}({', '.join(args_formatted)})")
    return f"<|tool_call_start|>[{', '.join(calls_str)}]<|tool_call_end|>"


class LfmMixin:
    """Mixin for LFM tool calling support."""

    def _customize_model_args(
        self,
        model_args: dict[str, Any],
        tools: list[ChatCompletionFunctionToolParam] | None,
        chat_log: conversation.ChatLog,
    ) -> None:
        """Customize model_args specifically for LFM model tool calling."""
        if not tools:
            return

        # 1. Remove top-level tools and tool_choice from model_args to prevent server-side JSON schema parsing
        model_args.pop("tools", None)
        model_args.pop("tool_choice", None)

        # 2. Add tools to extra_body["chat_template_kwargs"]["tools"]
        extra_body = model_args.setdefault("extra_body", {})
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        chat_template_kwargs["tools"] = tools

        # 3. Instruct backend servers (llama.cpp/vLLM) not to perform server-side tool calling
        extra_body["function_call"] = "none"

        # 4. Inject tool definitions into system message if messages are present
        messages = model_args.get("messages", [])
        tools_json = [tool["function"] for tool in tools if "function" in tool]
        tools_prompt = f"\n\nList of tools: {json.dumps(tools_json)}"

        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += tools_prompt
        elif messages:
            messages.insert(
                0, {"role": "system", "content": f"List of tools: {json.dumps(tools_json)}"}
            )

    async def _convert_content_to_chat_message(
        self,
        content: conversation.Content,
    ) -> ChatCompletionMessageParam | None:
        """Convert content to OpenAI Chat message, converting Assistant tool calls to LFM text representation."""
        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            lfm_tool_str = _format_lfm_tool_call_string(content.tool_calls)
            text_content = content.content or ""
            full_content = (
                f"{lfm_tool_str} {text_content}".strip()
                if text_content
                else lfm_tool_str
            )
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=full_content,
            )

        return await super()._convert_content_to_chat_message(content)

    async def _transform_stream(
        self,
        stream: AsyncStream[ChatCompletionChunk],
        strip_emojis: bool,
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
        """Transform LFM streaming response, detecting and parsing tool calls."""
        import asyncio
        import demoji
        from custom_components.local_openai.entity import _SUPPORTS_THINKING

        new_msg = True
        pending_think = ""
        in_think = False
        seen_visible = False
        loop = asyncio.get_running_loop()

        pending_lfm_tool_content = ""
        in_lfm_tool_call = False

        async for event in stream:
            chunk: conversation.AssistantContentDeltaDict = {}

            if not event.choices:
                continue

            choice = event.choices[0]
            delta = choice.delta

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
                    parts = content.split("<|tool_call_start|>", 1)
                    content = parts[1] if len(parts) > 1 else ""

                # Check for LFM tool call end token
                if "<|tool_call_end|>" in content:
                    in_lfm_tool_call = False
                    parts = content.split("<|tool_call_end|>", 1)
                    pending_lfm_tool_content += parts[0]

                    # Parse tool call
                    tool_calls_data = _parse_lfm_tool_calls(
                        pending_lfm_tool_content, None
                    )
                    if tool_calls_data:
                        chunk["tool_calls"] = [
                            llm.ToolInput(
                                id=f"call_{i}",
                                tool_name=tc["name"],
                                tool_args=tc["args"],
                            )
                            for i, tc in enumerate(tool_calls_data)
                        ]
                    pending_lfm_tool_content = ""
                    content = parts[1] if len(parts) > 1 else ""

                if in_lfm_tool_call:
                    pending_lfm_tool_content += content
                    content = ""

                # Handle think tags
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

                if seen_visible and content:
                    chunk["content"] = content

            if choice.finish_reason:
                try:
                    if event.timings:
                        self.extra_state_attributes = {"timings": event.timings}
                except Exception:
                    pass

            if (
                seen_visible
                or chunk.get("tool_calls")
                or chunk.get("role")
                or chunk.get("thinking_content")
            ):
                yield chunk


def get_conversation_config_schema() -> dict:
    """Return conversation config schema for LFM."""
    return {}


def get_ai_task_config_schema() -> dict:
    """Return AI task config schema for LFM."""
    return get_conversation_config_schema()


# Import these at bottom to avoid circular imports
from custom_components.local_openai.ai_task import LocalAITaskEntity  # noqa: E402
from custom_components.local_openai.conversation import (
    LocalAiConversationEntity,
)


class LfmConversationEntity(LfmMixin, LocalAiConversationEntity):
    """Conversation agent with LFM tool calling support."""


class LfmAITaskEntity(LfmMixin, LocalAITaskEntity):
    """AI Task entity with LFM tool calling support."""

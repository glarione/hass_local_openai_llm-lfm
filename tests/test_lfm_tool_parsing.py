"""Tests for LFM tool calling parser and mixin."""

import asyncio
from unittest import TestCase
from unittest.mock import MagicMock

from homeassistant.components import conversation
from homeassistant.helpers import llm

from custom_components.local_openai.entities.lfm import (
    LfmMixin,
    _format_lfm_tool_call_string,
    _parse_lfm_tool_calls,
)


class TestParseLfmToolCalls(TestCase):
    """Tests for _parse_lfm_tool_calls function using AST."""

    def test_parse_single_no_args(self):
        """Test parsing GetLiveContext() from error trace."""
        result = _parse_lfm_tool_calls("[GetLiveContext()]", {"GetLiveContext"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "GetLiveContext")
        self.assertEqual(result[0]["args"], {})

    def test_parse_single_keyword_args_single_quotes(self):
        """Test parsing HassTurnOn(name='TV LED', area='Living Room') with single quotes."""
        content = "[HassTurnOn(name='TV LED', area='Living Room')]"
        result = _parse_lfm_tool_calls(content, {"HassTurnOn"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "HassTurnOn")
        self.assertEqual(result[0]["args"], {"name": "TV LED", "area": "Living Room"})

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls inside list."""
        content = '[update_status(id="1"), get_info(id="2")]'
        result = _parse_lfm_tool_calls(content, {"update_status", "get_info"})
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "update_status")
        self.assertEqual(result[0]["args"], {"id": "1"})
        self.assertEqual(result[1]["name"], "get_info")
        self.assertEqual(result[1]["args"], {"id": "2"})

    def test_parse_disallowed_function(self):
        """Test filtering out disallowed functions."""
        content = '[allowed_func(arg="1"), disallowed_func(arg="2")]'
        result = _parse_lfm_tool_calls(content, {"allowed_func"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "allowed_func")

    def test_parse_empty_content(self):
        """Test empty string returns empty list."""
        self.assertEqual(_parse_lfm_tool_calls("", {"func"}), [])


class TestFormatLfmToolCallString(TestCase):
    """Tests for formatting ToolInput into LFM string representation."""

    def test_format_lfm_tool_call_string(self):
        tool_input = llm.ToolInput(
            id="call_0",
            tool_name="HassTurnOn",
            tool_args={"name": "TV LED", "area": "Living Room"},
        )
        formatted = _format_lfm_tool_call_string([tool_input])
        self.assertEqual(
            formatted,
            "<|tool_call_start|>[HassTurnOn(name='TV LED', area='Living Room')]<|tool_call_end|>",
        )


class DummyLfmEntity(LfmMixin):
    """Dummy class to test LfmMixin methods."""

    async def _convert_content_to_chat_message(self, content):
        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            return await super()._convert_content_to_chat_message(content)
        if isinstance(content, conversation.AssistantContent):
            return {"role": "assistant", "content": content.content}
        return None


class TestLfmMixin(TestCase):
    """Tests for LfmMixin behavior."""

    def test_convert_content_to_chat_message_with_tool_calls(self):
        """Test that AssistantContent with tool_calls is converted into LFM text format."""
        mixin = DummyLfmEntity()
        content = conversation.AssistantContent(
            agent_id="test_agent",
            content="Checking status...",
            tool_calls=[
                llm.ToolInput(
                    id="call_0",
                    tool_name="HassTurnOn",
                    tool_args={"name": "TV LED"},
                )
            ],
        )

        msg = asyncio.run(mixin._convert_content_to_chat_message(content))
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(
            msg["content"],
            "<|tool_call_start|>[HassTurnOn(name='TV LED')]<|tool_call_end|> Checking status...",
        )
        self.assertNotIn("tool_calls", msg)

    def test_customize_model_args(self):
        """Test model_args customization for LFM."""
        mixin = DummyLfmEntity()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "HassTurnOn",
                    "description": "Turn on light",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
            }
        ]
        model_args = {
            "model": "lfm-2.5",
            "tools": tools,
            "tool_choice": "auto",
            "messages": [{"role": "system", "content": "You are a helpful assistant."}],
        }
        chat_log = MagicMock()

        mixin._customize_model_args(model_args, tools, chat_log)

        # 1. Top level tools & tool_choice must be removed
        self.assertNotIn("tools", model_args)
        self.assertNotIn("tool_choice", model_args)

        # 2. Tools added to extra_body["chat_template_kwargs"]["tools"]
        self.assertEqual(
            model_args["extra_body"]["chat_template_kwargs"]["tools"], tools
        )

        # 3. function_call set to "none"
        self.assertEqual(model_args["extra_body"]["function_call"], "none")

        # 4. System prompt updated with List of tools
        self.assertIn("List of tools:", model_args["messages"][0]["content"])
        self.assertIn("HassTurnOn", model_args["messages"][0]["content"])


if __name__ == "__main__":
    import unittest

    unittest.main()

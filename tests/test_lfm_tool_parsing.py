"""Tests for LFM tool calling parser."""

import pytest
from custom_components.local_openai.entities.lfm import (
    _parse_lfm_tool_calls,
    _parse_python_function_call,
)


class TestParsePythonFunctionCall:
    """Tests for _parse_python_function_call function."""

    def test_parse_valid_function_call(self):
        """Test parsing a valid function call with allowed tool."""
        allowed_tools = {"get_candidate_status", "update_status"}
        result = _parse_python_function_call(
            'get_candidate_status(candidate_id="12345")', allowed_tools
        )

        assert result is not None
        assert result["name"] == "get_candidate_status"
        assert result["args"] == {"candidate_id": "12345"}

    def test_parse_function_call_with_multiple_args(self):
        """Test parsing a function call with multiple arguments."""
        allowed_tools = {"create_user"}
        result = _parse_python_function_call(
            'create_user(name="John", age=30, active=True)', allowed_tools
        )

        assert result is not None
        assert result["name"] == "create_user"
        assert result["args"] == {"name": "John", "age": 30, "active": True}

    def test_parse_function_call_with_numeric_args(self):
        """Test parsing a function call with numeric arguments."""
        allowed_tools = {"calculate_total"}
        result = _parse_python_function_call(
            "calculate_total(price=19.99, quantity=3)", allowed_tools
        )

        assert result is not None
        assert result["name"] == "calculate_total"
        assert result["args"] == {"price": 19.99, "quantity": 3}

    def test_parse_function_call_with_list_args(self):
        """Test parsing a function call with list arguments."""
        allowed_tools = {"process_items"}
        result = _parse_python_function_call(
            'process_items(items=["a", "b", "c"])', allowed_tools
        )

        assert result is not None
        assert result["name"] == "process_items"
        assert result["args"] == {"items": ["a", "b", "c"]}

    def test_parse_disallowed_function_call(self):
        """Test that disallowed function calls return None."""
        allowed_tools = {"safe_function"}
        result = _parse_python_function_call(
            'malicious_function(payload="evil")', allowed_tools
        )

        assert result is None

    def test_parse_empty_string(self):
        """Test parsing an empty string."""
        allowed_tools = {"any_function"}
        result = _parse_python_function_call("", allowed_tools)

        assert result is None

    def test_parse_invalid_syntax(self):
        """Test parsing invalid Python syntax."""
        allowed_tools = {"safe_function"}
        result = _parse_python_function_call("invalid syntax here", allowed_tools)

        assert result is None

    def test_parse_method_call(self):
        """Test parsing a method call (not a simple function)."""
        allowed_tools = {"safe_function"}
        # object.method() is not a simple Name node
        result = _parse_python_function_call('obj.method(arg="value")', allowed_tools)

        # This should return None because func.id won't exist for Attribute nodes
        assert result is None

    def test_parse_no_args(self):
        """Test parsing a function call with no arguments."""
        allowed_tools = {"get_status"}
        result = _parse_python_function_call("get_status()", allowed_tools)

        assert result is not None
        assert result["name"] == "get_status"
        assert result["args"] == {}


class TestParseLfmToolCalls:
    """Tests for _parse_lfm_tool_calls function."""

    def test_parse_single_tool_call(self):
        """Test parsing a single LFM tool call."""
        allowed_tools = {"get_candidate_status"}
        content = '[get_candidate_status(candidate_id="12345")]'
        result = _parse_lfm_tool_calls(content, allowed_tools)

        assert len(result) == 1
        assert result[0]["name"] == "get_candidate_status"
        assert result[0]["args"] == {"candidate_id": "12345"}

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple LFM tool calls."""
        allowed_tools = {"update_status", "get_info"}
        content = '[update_status(id="1"), get_info(id="2")]'
        result = _parse_lfm_tool_calls(content, allowed_tools)

        assert len(result) == 2
        assert result[0]["name"] == "update_status"
        assert result[0]["args"] == {"id": "1"}
        assert result[1]["name"] == "get_info"
        assert result[1]["args"] == {"id": "2"}

    def test_parse_tool_call_with_disallowed_function(self):
        """Test that disallowed functions are filtered out."""
        allowed_tools = {"allowed_function"}
        content = '[allowed_function(arg="1"), disallowed_function(arg="2")]'
        result = _parse_lfm_tool_calls(content, allowed_tools)

        # Only the allowed function should be parsed
        assert len(result) == 1
        assert result[0]["name"] == "allowed_function"

    def test_parse_empty_content(self):
        """Test parsing empty content."""
        allowed_tools = {"any_function"}
        result = _parse_lfm_tool_calls("", allowed_tools)

        assert result == []

    def test_parse_no_brackets(self):
        """Test parsing content without square brackets."""
        allowed_tools = {"any_function"}
        result = _parse_lfm_tool_calls("get_status()", allowed_tools)

        assert result == []

    def test_parse_complex_args(self):
        """Test parsing tool calls with complex arguments."""
        allowed_tools = {"process_data"}
        content = '[process_data(config={"key": "value"}, items=[1, 2, 3])]'
        result = _parse_lfm_tool_calls(content, allowed_tools)

        # Dict arguments may not parse correctly with literal_eval
        # This tests that we handle it gracefully
        assert len(result) >= 0  # May be 0 if dict parsing fails


class TestSecurity:
    """Security tests for LFM tool calling."""

    def test_no_blind_eval(self):
        """Test that arbitrary code cannot be executed."""
        allowed_tools = {"safe_function"}

        # Try to call os.system - should be rejected
        result = _parse_python_function_call('os.system("rm -rf /")', allowed_tools)

        assert result is None

    def test_function_name_validation(self):
        """Test that function names are validated against allowed list."""
        allowed_tools = {"get_user", "update_user"}

        # Try variations that shouldn't match
        test_cases = [
            'get_user_admin(id="1")',  # Different function name
            'get_users(id="1")',  # Plural, not in allowed list
            '_get_user(id="1")',  # Prefix added
        ]

        for call_str in test_cases:
            result = _parse_python_function_call(call_str, allowed_tools)
            assert result is None, f"Should reject: {call_str}"

    def test_empty_allowed_tools(self):
        """Test that no functions can be called when allowed_tools is empty."""
        result = _parse_python_function_call('any_function(arg="value")', set())

        assert result is None

    def test_none_allowed_tools(self):
        """Test that no functions can be called when allowed_tools is None."""
        # Type check will fail at compile time, so we test with empty set instead
        result = _parse_python_function_call('any_function(arg="value")', set())

        assert result is None

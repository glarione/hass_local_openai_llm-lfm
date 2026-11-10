"""Tests for per-subentry headers and the extended template context."""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import MagicMock, patch

from custom_components.local_openai.const import (
    CONF_CUSTOM_HEADERS,
    CONF_SERVER_HEADERS,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigEntryState, ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr

DOMAIN = "local_openai"


def _make_entry(headers: dict | None) -> ConfigEntry:
    """Config entry, optionally carrying entry-level headers."""
    entry = ConfigEntry(
        domain=DOMAIN,
        title="Test Server",
        data={
            CONF_MODEL: "test-model",
            **({CONF_CUSTOM_HEADERS: headers} if headers else {}),
        },
        source="user",
        state=ConfigEntryState.LOADED,
        version=1,
        minor_version=1,
        discovery_keys=MappingProxyType({}),
        options=None,
        subentries_data=None,
        unique_id=None,
    )
    entry.runtime_data = MagicMock()
    return entry


def _make_subentry(headers: dict | None) -> ConfigSubentry:
    """Conversation subentry, optionally carrying its own headers."""
    data = {CONF_MODEL: "test-model"}
    if headers is not None:
        data[CONF_CUSTOM_HEADERS] = headers
    return ConfigSubentry(
        subentry_id="sub-1",
        subentry_type="conversation",
        title="Conversation Agent",
        data=MappingProxyType(data),
        unique_id=None,
    )


def _make_entity(hass: HomeAssistant, entry: ConfigEntry, subentry: ConfigSubentry):
    """Build the conversation entity wired to a hass instance."""
    entry._subentries = {subentry.subentry_id: subentry}
    entity = LocalAiConversationEntity(entry, subentry)
    entity.hass = hass
    entity._attr_has_entity_name = False
    entity._attr_name = "Test"
    return entity


def _chat_log(device_id: str | None = None) -> MagicMock:
    """Chat log, optionally with an LLM context naming the asking device."""
    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.content = []
    chat_log.conversation_id = "conv-1"
    if device_id is None:
        chat_log.llm_api = None
    else:
        chat_log.llm_api = MagicMock()
        chat_log.llm_api.tools = []
        chat_log.llm_api.llm_context = MagicMock(
            device_id=device_id,
            language="ru",
            agent_id="conversation.test",
            context=MagicMock(user_id="user-1"),
        )
    return chat_log


async def _capture_headers(entity, chat_log) -> dict:
    """Run the entity's request build and return the extra headers it sends."""
    captured: dict = {}

    async def fake_loop(client, model_args, *args, **kwargs):
        captured.update(model_args)

    with patch.object(entity, "_run_agent_loop", side_effect=fake_loop):
        await entity._async_handle_chat_log(chat_log)

    return captured["extra_headers"]


class TestSubentryHeaders:
    """Headers on the subentry override the ones on the config entry."""

    async def test_subentry_headers_win_over_entry_headers(
        self, hass: HomeAssistant
    ) -> None:
        entry = _make_entry(
            {CONF_SERVER_HEADERS: [{"Key": "X-Session", "Value": "entry-level"}]}
        )
        subentry = _make_subentry(
            {CONF_SERVER_HEADERS: [{"Key": "X-Session", "Value": "subentry-level"}]}
        )
        entity = _make_entity(hass, entry, subentry)

        headers = await _capture_headers(entity, _chat_log())

        assert headers["X-Session"] == "subentry-level"

    async def test_entry_headers_still_apply_without_subentry_headers(
        self, hass: HomeAssistant
    ) -> None:
        """An agent that never set headers keeps the server-wide ones."""
        entry = _make_entry(
            {CONF_SERVER_HEADERS: [{"Key": "X-Server", "Value": "shared"}]}
        )
        entity = _make_entity(hass, entry, _make_subentry(None))

        headers = await _capture_headers(entity, _chat_log())

        assert headers["X-Server"] == "shared"

    async def test_empty_subentry_section_disables_entry_headers(
        self, hass: HomeAssistant
    ) -> None:
        """An explicit empty section means "no extra headers", not "inherit"."""
        entry = _make_entry(
            {CONF_SERVER_HEADERS: [{"Key": "X-Server", "Value": "shared"}]}
        )
        subentry = _make_subentry({CONF_SERVER_HEADERS: []})
        entity = _make_entity(hass, entry, subentry)

        headers = await _capture_headers(entity, _chat_log())

        assert "X-Server" not in headers


class TestExtendedTemplateContext:
    """device_id / area_id / language / user_id are renderable in headers."""

    async def test_device_area_language_and_user_render(
        self, hass: HomeAssistant
    ) -> None:
        area = ar.async_get(hass).async_create("Kitchen")
        entry = _make_entry(None)
        hass.config_entries._entries[entry.entry_id] = entry
        device = dr.async_get(hass).async_get_or_create(
            config_entry_id=entry.entry_id,
            identifiers={(DOMAIN, "sat-kitchen")},
        )
        dr.async_get(hass).async_update_device(device.id, area_id=area.id)

        subentry = _make_subentry(
            {
                CONF_SERVER_HEADERS: [
                    {
                        "Key": "X-Session",
                        "Value": "sat-{{ device_id }}-{{ area_id }}-{{ language }}-{{ user_id }}",
                    }
                ]
            }
        )
        entity = _make_entity(hass, entry, subentry)

        headers = await _capture_headers(entity, _chat_log(device.id))

        assert headers["X-Session"] == f"sat-{device.id}-{area.id}-ru-user-1"

    async def test_missing_context_renders_empty_not_raises(
        self, hass: HomeAssistant
    ) -> None:
        """No satellite context (e.g. a text chat) must not break the request."""
        entry = _make_entry(None)
        subentry = _make_subentry(
            {CONF_SERVER_HEADERS: [{"Key": "X-Device", "Value": "{{ device_id }}"}]}
        )
        entity = _make_entity(hass, entry, subentry)

        headers = await _capture_headers(entity, _chat_log())

        assert headers["X-Device"] == ""

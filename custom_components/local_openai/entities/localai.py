"""Server-specific entities for LocalAI."""

from __future__ import annotations

from typing import Any

from custom_components.local_openai.ai_task import LocalAITaskEntity
from custom_components.local_openai.conversation import LocalAiConversationEntity


class LocalAIServerMixin:
    """Mixin for LocalAI server entities with shared logic.

    LocalAI does not read a top-level ``chat_template_kwargs`` field the way
    llama.cpp's server does. Chat template variables are supplied via the
    OpenAI ``metadata`` field, with string values.

    See https://github.com/mudler/LocalAI/pull/10359
    """

    _chat_template_kwargs_key = "metadata"

    # noinspection PyMethodMayBeStatic
    def _format_chat_template_kwarg(self, value: Any) -> Any:
        """LocalAI expects metadata values as strings."""
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)


class LocalAIServerConversationEntity(
    LocalAIServerMixin,
    LocalAiConversationEntity,
):
    """Conversation agent for LocalAI servers."""


class LocalAIServerAITaskEntity(LocalAIServerMixin, LocalAITaskEntity):
    """AI Task entity for LocalAI servers."""
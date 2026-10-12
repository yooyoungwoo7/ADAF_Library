"""Natural-language system definition (LLM assistant) requests and responses."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .common import SystemSpec, TimeSpan

Provider = Literal["anthropic", "openai"]


class AssistMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class AssistForwardRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    provider: Optional[Provider] = None
    # what is on screen now, so "set mu to 3" edits it instead of starting over
    current_system: Optional[SystemSpec] = None
    current_time: Optional[TimeSpan] = None
    # earlier turns of this chat (oldest first)
    history: List[AssistMessage] = Field(default_factory=list, max_length=20)


class AssistForwardResponse(BaseModel):
    system: SystemSpec
    time: TimeSpan
    reply: str
    provider: Provider
    model: str
    attempts: int


class AssistProviderStatus(BaseModel):
    label: str
    configured: bool
    source: Optional[Literal["env", "settings"]] = None   # where the key comes from
    key_hint: Optional[str] = None                         # last 4 characters only
    model: str
    default_model: str


class AssistStatus(BaseModel):
    provider: Optional[Provider] = None                    # the one used when a request names none
    providers: Dict[str, AssistProviderStatus]
    settings_file: str


class AssistSettingsUpdate(BaseModel):
    provider: Provider
    api_key: Optional[str] = Field(None, max_length=500)  # None keeps the stored key, "" removes it
    model: Optional[str] = Field(None, max_length=100)    # None keeps it, "" restores the default
    make_default: bool = True

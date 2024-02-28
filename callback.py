"""Callback handlers used in the app."""
import sys
from typing import Any, Dict, List

from langchain.callbacks.base import AsyncCallbackHandler

from schemas import ChatResponse

from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.schema import LLMResult
from uuid import UUID


class StreamingLLMCallbackHandler(FinalStreamingStdOutCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        super().__init__()
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.append_to_last_tokens(token)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.check_if_answer_reached():
            self.answer_reached = True
            if self.stream_prefix:
                for t in self.last_tokens:
                    sys.stdout.write(t)
                sys.stdout.flush()
            return

        # ... if yes, then print tokens from now on
        if self.answer_reached:
            resp = ChatResponse(sender="bot", message=token, type="stream")
            await self.websocket.send_json(resp.dict())
            # sys.stdout.write(token)
            # sys.stdout.flush()


class QuestionGenCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        resp = ChatResponse(
            sender="bot", message="Melakukan sintesis pertanyaan...", type="info"
        )
        await self.websocket.send_json(resp.dict())


class CustomStreamingStdOutCallbackHandler(FinalStreamingStdOutCallbackHandler):
    buffer: List[Tuple[str, float]] = []
    stop_token = "#!stop!#"

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        super().on_llm_start(serialized, prompts, **kwargs)
        self.buffer = []

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self.add_to_buffer(self.stop_token)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.last_tokens.append(token)
        if len(self.last_tokens) > len(self.answer_prefix_tokens):
            self.last_tokens.pop(0)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.last_tokens == self.answer_prefix_tokens:
            self.answer_reached = True
            # Do not print the last token in answer_prefix_tokens,
            # as it's not part of the answer yet
            return

        # ... if yes, then append tokens to buffer
        if self.answer_reached:
            self.add_to_buffer(token)

    def add_to_buffer(self, token: str) -> None:
        now = datetime.now()
        self.buffer.append((token, now))

    def stream_chars(self):
        while True:
            # when we didn't receive any token yet, just continue
            if len(self.buffer) == 0:
                continue

            token, timestamp = self.buffer.pop(0)

            if token != self.stop_token:
                for character in token:
                    yield (character, timestamp)
                    time.sleep(
                        0.2
                    )  # Remove this line. It's just for illustration purposes
            else:
                break

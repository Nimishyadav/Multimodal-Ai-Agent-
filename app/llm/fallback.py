"""Try compatible backup models when a model request fails."""
import time
from app.multi_agent.errors import AgentExecutionError


class ProviderOutputError(ValueError):
    """Provider ignored a required tool call or returned no usable result."""


def can_fail_over(error):
    if isinstance(error, ProviderOutputError):
        return True
    status = getattr(error, 'status_code', None) or getattr(error, 'code', None)
    cause = getattr(error, '__cause__', None)
    if status is None and cause is not None:
        status = getattr(cause, 'status_code', None) or getattr(cause, 'code', None)
    body = getattr(error, 'body', {}) or {}
    detail = body.get('error', body) if isinstance(body, dict) else {}
    if status == 400 and isinstance(detail, dict) and detail.get('code') == 'tool_use_failed':
        return True
    if status in {401, 403, 404, 408, 429, 500, 502, 503, 504}:
        return True
    return isinstance(error, (TimeoutError, ConnectionError)) or type(error).__name__ in {
        'APIConnectionError', 'APITimeoutError', 'ConnectError', 'ConnectTimeout', 'ReadTimeout',
        'ServiceUnavailable', 'ResourceExhausted', 'TooManyRequests'}


class FallbackModel:
    def __init__(self, models, shared=None, tools=None, tool_options=None):
        if not models:
            raise ValueError('No compatible model provider is configured.')
        self.models = models
        self.shared = shared if shared is not None else {'cooldown': {}, 'events': []}
        self.tools = tools
        self.tool_options = tool_options or {}

    def bind_tools(self, tools, **kwargs):
        return FallbackModel(self.models, self.shared, list(tools), kwargs)

    def model_copy(self, update=None):
        return FallbackModel([(name, model.model_copy(update=update or {})) for name,model in self.models],
                             self.shared, self.tools, self.tool_options)

    def drain_events(self):
        events = self.shared['events'][:]
        self.shared['events'].clear()
        return events

    def invoke(self, messages):
        return self.invoke_with_budget(messages, lambda: None)

    def invoke_with_budget(self, messages, extra_attempt):
        tried = 0
        for name, model in self.models:
            if self.shared['cooldown'].get(name, 0) > time.monotonic():
                continue
            if tried:
                extra_attempt()
            tried += 1
            try:
                active = model.bind_tools(self.tools, **self.tool_options) if self.tools is not None else model
                response = active.invoke(messages)
                # Convert Gemini text blocks to the plain text expected by the rest of the app.
                if isinstance(getattr(response, 'content', None), list):
                    text = "\n".join(block.get('text', '') for block in response.content
                                     if isinstance(block, dict) and block.get('type') == 'text')
                    response = response.model_copy(update={'content': text})
                calls = getattr(response, 'tool_calls', [])
                forced = self.tool_options.get('tool_choice') not in (None, 'auto', 'none')
                if getattr(response, 'invalid_tool_calls', []) or (forced and not calls) or (self.tool_options.get('tool_choice') == 'none' and calls):
                    raise ProviderOutputError('Required tool output was missing or malformed.')
                if not calls and not str(getattr(response, 'content', '')).strip():
                    raise ProviderOutputError('No usable model output.')
                self.shared['events'].append('Model provider used: ' + name)
                return response
            except Exception as error:
                if not can_fail_over(error):
                    raise
                self.shared['cooldown'][name] = time.monotonic() + 60
                self.shared['events'].append(name + ' unavailable; trying configured backup.')
        raise AgentExecutionError('All configured compatible model APIs are unavailable or cooling down. '
                                  'Check quotas/keys, configure a backup in .env, or wait at least 60 seconds.') from None

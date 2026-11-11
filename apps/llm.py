"""One place that builds every chat model the service uses.

The provider is any OpenAI-compatible gateway (modelapi.vn by default):
    LLM_BASE_URL   https://modelapi.vn/v1
    LLM_API_KEY    sk-...
    LLM_MODEL      main model (agents, RAG answers, career review)
    LLM_FAST_MODEL cheap model (scope classification)
If LLM_API_KEY is empty but GROQ_API_KEY is set, Groq's OpenAI-compatible endpoint is used instead.
"""
import os

DEFAULT_BASE_URL = 'https://modelapi.vn/v1'
DEFAULT_MODEL = 'gpt-6.1-terra'
DEFAULT_FAST_MODEL = 'gpt-6-luna'
GROQ_BASE_URL = 'https://api.groq.com/openai/v1'


class LlmNotConfigured(RuntimeError):
    pass


def llm_settings(fast: bool = False) -> dict:
    api_key = os.getenv('LLM_API_KEY', '').strip()
    if api_key:
        base_url = os.getenv('LLM_BASE_URL', '').strip() or DEFAULT_BASE_URL
        model = os.getenv('LLM_FAST_MODEL' if fast else 'LLM_MODEL', '').strip()
        model = model or (DEFAULT_FAST_MODEL if fast else DEFAULT_MODEL)
        return {'api_key': api_key, 'base_url': base_url.rstrip('/'), 'model': model, 'provider': 'openai-compatible'}

    groq_key = os.getenv('GROQ_API_KEY', '').strip()
    if groq_key:
        model = os.getenv('GROQ_FAST_MODEL' if fast else 'GROQ_MODEL', '').strip()
        model = model or ('openai/gpt-oss-20b' if fast else 'openai/gpt-oss-120b')
        return {'api_key': groq_key, 'base_url': GROQ_BASE_URL, 'model': model, 'provider': 'groq'}

    raise LlmNotConfigured('Chưa cấu hình LLM_API_KEY cho dịch vụ AI.')


def model_name(fast: bool = False) -> str:
    try:
        return llm_settings(fast)['model']
    except LlmNotConfigured:
        return 'unconfigured'


def build_chat_model(fast: bool = False, **overrides):
    """ChatOpenAI pointed at the configured gateway. Raises LlmNotConfigured without a key."""
    from langchain_openai import ChatOpenAI

    settings = llm_settings(fast)
    options = {
        'temperature': 0,
        'timeout': 25 if fast else 60,
        'max_retries': 1 if fast else 2,
    }
    options.update(overrides)
    return ChatOpenAI(api_key=settings['api_key'], base_url=settings['base_url'], model=settings['model'], **options)

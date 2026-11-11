"""Tool-calling loop shared by the member and staff assistants."""
import asyncio
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

SECRET_PATTERNS = [
    re.compile(r'\bsk-[A-Za-z0-9_\-]{16,}'),
    re.compile(r'\bgsk_[A-Za-z0-9]{16,}'),
    re.compile(r'Bearer\s+[A-Za-z0-9\-_.=]{20,}', re.IGNORECASE),
    re.compile(r'eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}'),
    re.compile(r'mongodb(\+srv)?://\S+', re.IGNORECASE),
    re.compile(r'amqps?://\S+', re.IGNORECASE),
]

INJECTION_PATTERNS = [
    r'ignore (all|previous|the above)', r'bỏ qua (mọi|tất cả|các)? ?(hướng dẫn|chỉ dẫn|quy tắc)',
    r'system prompt', r'(tiết lộ|cho xem|in ra).{0,30}(prompt|token|mật khẩu|api ?key|cấu hình)',
    r'api[ _-]?key', r'connection string', r'jailbreak', r'developer mode', r'\bDAN\b',
]


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub('[đã ẩn]', text)
    return text


def looks_like_injection(message: str) -> bool:
    lower = message.lower()
    return any(re.search(pattern, lower if pattern != r'\bDAN\b' else message) for pattern in INJECTION_PATTERNS)


def history_messages(history: list[dict] | None, max_turns: int = 10) -> list:
    messages = []
    for turn in (history or [])[-max_turns:]:
        content = str(turn.get('content') or '').strip()[:1500]
        if not content:
            continue
        messages.append(HumanMessage(content=content) if turn.get('role') == 'user' else AIMessage(content=content))
    return messages


async def run_agent(llm, system_prompt: str, message: str, tools: list, history: list[dict] | None = None,
                    max_tool_calls: int = 6) -> dict:
    """Runs the model with an allow-listed tool box. Returns {message, toolTrace, toolActions}."""
    bound = llm.bind_tools(tools) if tools else llm
    tools_by_name = {item.name: item for item in tools}
    messages = [SystemMessage(content=system_prompt), *history_messages(history), HumanMessage(content=message)]
    trace, tool_actions = [], []
    calls = 0

    while True:
        result = await asyncio.to_thread(bound.invoke, messages)
        messages.append(result)
        tool_calls = getattr(result, 'tool_calls', None) or []
        if not tool_calls:
            text = result.content if isinstance(result.content, str) else ' '.join(
                part.get('text', '') for part in result.content if isinstance(part, dict))
            return {'message': redact(text.strip()), 'toolTrace': trace, 'toolActions': tool_actions}
        if calls >= max_tool_calls:
            return {'message': 'Yêu cầu cần quá nhiều bước. Bạn hãy hỏi cụ thể hơn hoặc chia nhỏ câu hỏi nhé.',
                    'toolTrace': trace, 'toolActions': tool_actions}

        for call in tool_calls:
            name = call.get('name', '')
            if calls >= max_tool_calls:
                output = {'ok': False, 'error': 'Đã đạt giới hạn số lần tra cứu cho một câu hỏi.'}
            else:
                calls += 1
                selected = tools_by_name.get(name)
                if not selected:
                    output = {'ok': False, 'error': 'Công cụ không nằm trong danh sách cho phép.'}
                else:
                    try:
                        output = await selected.ainvoke(call.get('args') or {})
                    except Exception as error:
                        output = {'ok': False, 'error': f'Tra cứu thất bại: {type(error).__name__}'}
            if not isinstance(output, dict):
                output = {'ok': True, 'data': output}
            trace.append({'tool': name, 'ok': bool(output.get('ok'))})
            if output.get('action'):
                tool_actions.append(output['action'])
            messages.append(ToolMessage(content=json.dumps(output, ensure_ascii=False, default=str)[:12000],
                                        tool_call_id=call.get('id') or name))

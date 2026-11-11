"""Calls the BFF with the caller's own bearer token, so every tool sees exactly
what the signed-in user may see through the website. No direct database access."""
import asyncio
import os

import requests


class BffClient:
    def __init__(self, authorization: str | None):
        self.authorization = (authorization or '').strip()
        self.base_url = os.getenv('BFF_INTERNAL_URL', 'http://localhost:3300/api/v2').rstrip('/')

    async def get(self, path: str, params: dict | None = None, auth: bool = True) -> dict:
        """Returns {'ok': True, 'data': ...} or {'ok': False, 'error': ...}; never raises."""
        if auth and not self.authorization.lower().startswith('bearer '):
            return {'ok': False, 'error': 'Thiếu phiên đăng nhập hợp lệ.'}
        headers = {'X-AI-Tool-Request': '1'}
        if auth:
            headers['Authorization'] = self.authorization
        clean_params = {key: value for key, value in (params or {}).items() if value not in (None, '')}

        def request_data():
            response = requests.get(f'{self.base_url}{path}', params=clean_params, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json()

        try:
            payload = await asyncio.to_thread(request_data)
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else 502
            if status in (401, 403):
                return {'ok': False, 'error': 'Tài khoản hiện tại không có quyền xem dữ liệu này.'}
            if status == 404:
                return {'ok': False, 'error': 'Không tìm thấy dữ liệu.'}
            return {'ok': False, 'error': f'API nghiệp vụ trả lỗi HTTP {status}.'}
        except Exception as error:
            return {'ok': False, 'error': f'Không thể kết nối API nghiệp vụ: {type(error).__name__}'}
        data = payload.get('data', payload) if isinstance(payload, dict) else payload
        return {'ok': True, 'data': data}


def unwrap(data):
    """BFF responses are sometimes wrapped twice ({data: {data: ...}})."""
    while isinstance(data, dict) and set(data) <= {'data', 'message', 'statusCode', 'success', 'processId', 'duration', 'code'} and 'data' in data:
        data = data['data']
    return data


def as_list(data) -> list:
    data = unwrap(data)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('result', 'items', 'data', 'docs', 'rows'):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def short(value, limit: int = 300) -> str:
    text = str(value or '').strip()
    return text if len(text) <= limit else text[:limit].rstrip() + '…'


def public_expert(user: dict) -> dict:
    """Only what the public expert page already shows. Never email, phone or account data."""
    profile = user.get('expertProfile') or {}
    expert_id = str(user.get('id') or user.get('_id') or '')
    return {
        'id': expert_id,
        'name': short(user.get('name'), 120),
        'major': short(profile.get('major') or user.get('major'), 160),
        'teachAt': short(profile.get('teachAt') or user.get('teachAt'), 160),
        'level': short(profile.get('level') or user.get('level'), 80),
        'price': profile.get('price', user.get('price')),
        'about': short(profile.get('information') or user.get('information'), 280),
        'avatar': user.get('fileAvartarUrl') or user.get('avatar'),
        'profileLink': f'/experts/{expert_id}' if expert_id else None,
        'bookingLink': f'/choose-schedule?expert_id={expert_id}' if expert_id else None,
    }

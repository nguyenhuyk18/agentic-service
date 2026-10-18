from langchain_core.tools import tool
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import os
import re
import requests

load_dotenv(Path(__file__).resolve().parent.parent / '.env')


class tools_func:
    """Allow-listed MajorGuide tools. All data access goes through the BFF."""

    def __init__(self, authorization: str | None = None):
        self.authorization = authorization or ''
        # Bind methods before creating schemas so self is never an LLM argument.
        for name in ('dashboard_summary', 'all_expert', 'search_users',
                     'pending_schedule_registrations', 'draft_email', 'accounting_summary'):
            setattr(self, name, tool(getattr(self, name)))
        self.bff_url = os.getenv('BFF_INTERNAL_URL', 'http://localhost:3300/api/v2').rstrip('/')

    async def _get(self, path: str, params: dict | None = None) -> dict:
        if not self.authorization.lower().startswith('bearer '):
            return {'ok': False, 'error': 'Thiếu phiên đăng nhập admin hợp lệ.'}

        def request_data():
            response = requests.get(
                f'{self.bff_url}{path}',
                params=params or {},
                headers={'Authorization': self.authorization, 'X-AI-Tool-Request': '1'},
                timeout=20,
            )
            response.raise_for_status()
            return response.json()

        try:
            payload = await asyncio.to_thread(request_data)
            return {'ok': True, 'data': payload.get('data', payload)}
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else 502
            return {'ok': False, 'error': f'API nghiệp vụ trả lỗi HTTP {status}.'}
        except Exception as error:
            return {'ok': False, 'error': f'Không thể kết nối API nghiệp vụ: {type(error).__name__}'}

    async def dashboard_summary(self) -> dict:
        """Lấy số liệu dashboard thật: booking, doanh thu, trạng thái và chuyên gia nổi bật."""
        return await self._get('/book/dashboard')

    async def all_expert(self, query: str = '', limit: int = 10) -> dict:
        """Tìm danh sách chuyên gia thật. query là tên cần tìm; limit từ 1 đến 20."""
        safe_limit = max(1, min(int(limit), 20))
        return await self._get('/user', {'role': 'expert', 'name': query[:100], 'page': 1, 'limit': safe_limit})

    async def search_users(self, query: str = '', role: str = '', status: str = '', page: int = 1, limit: int = 10) -> dict:
        """Tìm người dùng MajorGuide theo tên, role member/expert/admin và trạng thái tài khoản."""
        safe_role = role if role in {'member', 'expert', 'admin', ''} else ''
        return await self._get('/user', {
            'name': query[:100], 'role': safe_role, 'status': status[:30],
            'page': max(1, int(page)), 'limit': max(1, min(int(limit), 20)),
        })

    async def pending_schedule_registrations(self, status: str = '', page: int = 1) -> dict:
        """Lấy danh sách đơn đăng ký lịch tư vấn của chuyên gia đang chờ hoặc theo trạng thái yêu cầu."""
        return await self._get('/register', {'status': status[:30], 'page': max(1, int(page))})

    async def draft_email(self, email1: str, subject: str, content: str) -> dict:
        """Chỉ tạo bản nháp email cho đúng một người nhận. Không gửi email và cần admin xác nhận."""
        email = email1.strip().lower()
        if not re.fullmatch(r'[^\s@,;<>]+@[^\s@,;<>]+\.[^\s@,;<>]+', email):
            return {'ok': False, 'error': 'Địa chỉ email không hợp lệ.'}
        if not subject.strip() or not content.strip():
            return {'ok': False, 'error': 'Cần cung cấp đủ tiêu đề và nội dung email.'}
        if len(subject) > 150 or len(content) > 5000:
            return {'ok': False, 'error': 'Tiêu đề hoặc nội dung email vượt giới hạn.'}
        return {
            'ok': True,
            'action': {
                'type': 'email_draft', 'risk': 'write', 'requires_confirmation': True,
                'recipient': email, 'subject': subject.strip(), 'content': content.strip(),
            },
            'message': 'Đã tạo bản nháp. Email chưa được gửi và cần admin xác nhận.',
        }

    async def accounting_summary(self, start: str = '', end: str = '') -> dict:
        """Xem sổ thu chi: tổng thu/chi, tiền thu từ booking (booking.gross), doanh thu phí nền tảng
        (booking.platformRevenue), khoản phải trả chuyên gia (booking.expertPayable) và lợi nhuận (profit).
        Doanh thu thật của MajorGuide là platformRevenue, không phải tổng tiền thu. start/end có dạng YYYY-MM-DD."""
        params = {key: value for key, value in {'start': start, 'end': end}.items() if value}
        return await self._get('/accounting/summary', params)

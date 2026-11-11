from langchain_core.tools import tool
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import os
import re
import requests

load_dotenv(Path(__file__).resolve().parent.parent / '.env')


ADMIN_TOOLS = ('dashboard_summary', 'booking_overview', 'all_expert', 'search_users',
               'pending_schedule_registrations', 'notifications', 'draft_email', 'accounting_summary', 'site_info')
EXPERT_TOOLS = ('dashboard_summary', 'booking_overview', 'my_schedule_registration', 'notifications', 'site_info')


class tools_func:
    """Allow-listed MajorGuide tools. All data access goes through the BFF with the caller's token,
    so an expert only ever sees their own bookings even if the model asks for more."""

    def __init__(self, authorization: str | None = None, role: str = 'admin', user_id: str | None = None):
        self.authorization = authorization or ''
        # Unknown roles get the smaller toolbox; the BFF still enforces access per token.
        self.role = role if role in ('admin', 'expert') else 'expert'
        self.user_id = (user_id or '').strip()
        # Bind methods before creating schemas so self is never an LLM argument.
        for name in set(ADMIN_TOOLS + EXPERT_TOOLS):
            setattr(self, name, tool(getattr(self, name)))
        self.bff_url = os.getenv('BFF_INTERNAL_URL', 'http://localhost:3300/api/v2').rstrip('/')

    def tools(self) -> list:
        names = EXPERT_TOOLS if self.role == 'expert' else ADMIN_TOOLS
        return [getattr(self, name) for name in names]

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
        """Lấy số liệu dashboard thật: tổng booking, đã thanh toán, đang chờ, đã huỷ, doanh thu theo tháng,
        booking mới nhất và chuyên gia nổi bật (với chuyên gia: chỉ số liệu của chính họ)."""
        return await self._get('/book/dashboard')

    async def booking_overview(self, scope: str = 'upcoming', limit: int = 10) -> dict:
        """Danh sách lịch tư vấn. scope='upcoming': lịch đã thanh toán sắp diễn ra (từ hôm nay, sớm nhất trước);
        scope='recent': lịch vừa được đặt gần đây nhất (mọi trạng thái, mới nhất trước). limit 1-20.
        Chuyên gia chỉ thấy lịch của chính mình."""
        return await self._get('/book/ai-overview', {
            'scope': 'recent' if scope == 'recent' else 'upcoming',
            'limit': max(1, min(int(limit or 10), 20)),
        })

    async def my_schedule_registration(self) -> dict:
        """Các ca trực/đăng ký lịch tư vấn đang có hiệu lực trong tuần này của chính chuyên gia."""
        if not self.user_id:
            return {'ok': False, 'error': 'Thiếu thông tin chuyên gia.'}
        return await self._get(f'/register/expert/{self.user_id}')

    async def notifications(self, unread_only: bool = True, limit: int = 10) -> dict:
        """Thông báo gần đây của tài khoản (booking mới, thanh toán, đánh giá...). unread_only=True chỉ lấy chưa đọc."""
        return await self._get('/notifications', {
            'status': 'unread' if unread_only else '', 'page': 1, 'limit': max(1, min(int(limit or 10), 20)),
        })

    async def site_info(self, question: str) -> dict:
        """Tra cứu thông tin công khai về website MajorGuide (quy trình đặt lịch, thanh toán, bài test, liên hệ...)."""
        from apps.agents import site_knowledge
        return {'ok': True, 'data': site_knowledge.search(question[:500])}

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

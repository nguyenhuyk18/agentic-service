"""Tools for the member (student) assistant. Each instance is bound to one signed-in member."""
import asyncio
from datetime import date, datetime, timedelta, timezone
import re

from langchain_core.tools import tool

from . import site_knowledge
from .bff import BffClient, as_list, public_expert, short, unwrap

VN = timezone(timedelta(hours=7))
WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
WEEKDAY_VI = {'monday': 'Thứ Hai', 'tuesday': 'Thứ Ba', 'wednesday': 'Thứ Tư', 'thursday': 'Thứ Năm',
              'friday': 'Thứ Sáu', 'saturday': 'Thứ Bảy', 'sunday': 'Chủ Nhật'}
OBJECT_ID = re.compile(r'^[a-f0-9]{24}$')
MAX_BOOKING_DAYS_AHEAD = 7


def vietnam_today() -> date:
    return datetime.now(VN).date()


def _summarize_assessment(session: dict) -> dict:
    result = session.get('result') or {}
    groups = []
    for item in (result.get('topGroups') or [])[:5]:
        groups.append({
            'rank': item.get('rank'),
            'code': item.get('code'),
            'name': item.get('name') or item.get('title'),
            'matchScore': item.get('matchScore'),
            'fitReasons': [short(reason, 200) for reason in (item.get('fitReasons') or [])[:3]],
            'relatedMajors': [short(major, 120) for major in (item.get('relatedMajors') or [])[:6]],
        })
    profile = result.get('personalProfile') or {}
    explanation = result.get('explanation')
    summary = profile.get('summary') or (explanation.get('summary') if isinstance(explanation, dict) else explanation)
    return {
        'assessmentId': session.get('id'),
        'completedAt': session.get('completedAt') or session.get('updatedAt') or session.get('createdAt'),
        'riasecCode': result.get('riasecCode') or (result.get('mlModel') or {}).get('riasecCode'),
        'topGroups': groups,
        'strengths': (result.get('strengths') or [])[:5],
        'evidenceLevel': result.get('evidenceLevel'),
        'uncertainty': result.get('uncertainty'),
        'summary': short(summary, 600),
        'matchMeaning': result.get('matchMeaning'),
        'resultLink': '/quiz-history',
    }


class MemberTools:
    NAMES = ('site_info', 'latest_test_result', 'recommend_experts', 'search_experts',
             'expert_availability', 'my_bookings', 'career_knowledge')

    def __init__(self, authorization: str | None, user_id: str | None, career_search=None, client: BffClient | None = None):
        self.client = client or BffClient(authorization)
        self.user_id = (user_id or '').strip()
        self.career_search = career_search
        self.actions: list[dict] = []
        for name in self.NAMES:
            setattr(self, name, tool(getattr(self, name)))

    def tools(self) -> list:
        return [getattr(self, name) for name in self.NAMES]

    def _add_action(self, action: dict):
        if len(self.actions) < 6 and action not in self.actions:
            self.actions.append(action)

    async def site_info(self, question: str) -> dict:
        """Tra cứu thông tin công khai về website MajorGuide: tính năng, cách làm bài test, cách đặt lịch,
        thanh toán và phí, xem/tham gia buổi tư vấn, huỷ lịch, nhắn tin, blog, tài khoản, liên hệ hỗ trợ.
        question là câu hỏi của người dùng."""
        return {'ok': True, 'data': site_knowledge.search(question[:500])}

    async def _latest_completed_assessment(self) -> dict:
        history = await self.client.get('/career-assessments/me/history')
        if not history['ok']:
            return history
        sessions = [item for item in as_list(history['data']) if isinstance(item, dict) and item.get('status') == 'completed' and item.get('result')]
        if not sessions:
            return {'ok': True, 'data': None}
        sessions.sort(key=lambda item: str(item.get('completedAt') or item.get('updatedAt') or item.get('createdAt') or ''), reverse=True)
        return {'ok': True, 'data': sessions[0]}

    async def latest_test_result(self) -> dict:
        """Lấy kết quả bài kiểm tra định hướng nghề nghiệp GẦN NHẤT của chính người dùng đang chat:
        nhóm ngành phù hợp nhất, mức tương hợp, điểm mạnh, ngành học liên quan."""
        latest = await self._latest_completed_assessment()
        if not latest['ok']:
            return latest
        if not latest['data']:
            self._add_action({'type': 'link', 'label': 'Làm bài kiểm tra', 'href': '/quiz'})
            return {'ok': True, 'data': None, 'message': 'Người dùng chưa hoàn thành bài kiểm tra định hướng nào.'}
        summary = _summarize_assessment(latest['data'])
        self._add_action({'type': 'link', 'label': 'Xem kết quả bài test', 'href': '/quiz-history'})
        return {'ok': True, 'data': summary}

    async def recommend_experts(self, career_groups: str = '', limit: int = 4) -> dict:
        """Gợi ý chuyên gia/giảng viên phù hợp. career_groups là các mã nhóm ngành cách nhau bởi dấu phẩy
        (lấy từ latest_test_result); để trống thì tự dùng kết quả bài test gần nhất của người dùng."""
        codes = [code.strip() for code in career_groups.split(',') if code.strip()][:5]
        if not codes:
            latest = await self._latest_completed_assessment()
            if not latest['ok']:
                return latest
            if not latest['data']:
                return {'ok': True, 'data': [], 'message': 'Chưa có bài test để ghép chuyên gia. Hãy dùng search_experts theo ngành người dùng nêu, hoặc mời họ làm bài test.'}
            codes = [item.get('code') for item in (latest['data'].get('result') or {}).get('topGroups', []) if item.get('code')][:5]
        safe_limit = max(1, min(int(limit or 4), 6))
        response = await self.client.get('/career-assessments/experts', {'families': ','.join(codes), 'limit': safe_limit})
        if not response['ok']:
            return response
        experts = []
        for item in as_list(response['data'])[:safe_limit]:
            expert = public_expert(item)
            expert['matchReason'] = short(item.get('reason'), 240)
            expert['matchedGroup'] = (item.get('matchedGroup') or {}).get('name')
            experts.append(expert)
        if experts:
            self._add_action({'type': 'experts', 'experts': experts})
        return {'ok': True, 'data': experts}

    async def search_experts(self, query: str = '', limit: int = 5) -> dict:
        """Tìm chuyên gia đang hoạt động theo tên hoặc từ khoá chuyên ngành/nơi công tác (ví dụ "Kim", "kinh tế", "luật")."""
        safe_limit = max(1, min(int(limit or 5), 8))
        keyword = query.strip()[:100]
        response = await self.client.get('/user', {'role': 'expert', 'status': 'active', 'name': keyword, 'page': 1, 'limit': safe_limit}, auth=False)
        if not response['ok']:
            return response
        experts = [public_expert(item) for item in as_list(response['data'])]
        if not experts and keyword:
            # The API only matches names; fall back to matching the public profile text.
            everyone = await self.client.get('/user', {'role': 'expert', 'status': 'active', 'page': 1, 'limit': 50}, auth=False)
            if everyone['ok']:
                needle = site_knowledge.normalize(keyword)
                experts = [public_expert(item) for item in as_list(everyone['data'])
                           if needle in site_knowledge.normalize(' '.join(str(value) for value in public_expert(item).values() if isinstance(value, str)))]
        experts = experts[:safe_limit]
        if experts:
            self._add_action({'type': 'experts', 'experts': experts})
        return {'ok': True, 'data': experts}

    async def expert_availability(self, expert_id: str, day: str = '') -> dict:
        """Xem các ca tư vấn CÒN TRỐNG của một chuyên gia trong 7 ngày tới (đặt từ ngày mai) để hỗ trợ đặt lịch.
        expert_id lấy từ kết quả search_experts/recommend_experts; day (YYYY-MM-DD) để lọc một ngày, có thể bỏ trống."""
        expert_id = expert_id.strip()
        if not OBJECT_ID.match(expert_id):
            return {'ok': False, 'error': 'expert_id không hợp lệ; hãy tìm chuyên gia trước bằng search_experts.'}
        today = vietnam_today()
        candidates = [today + timedelta(days=offset) for offset in range(1, MAX_BOOKING_DAYS_AHEAD + 1)]
        if day:
            try:
                wanted = date.fromisoformat(day.strip()[:10])
            except ValueError:
                return {'ok': False, 'error': 'Ngày phải có dạng YYYY-MM-DD.'}
            if wanted not in candidates:
                return {'ok': True, 'data': [], 'message': f'Chỉ đặt được lịch từ {candidates[0].isoformat()} đến {candidates[-1].isoformat()}.'}
            candidates = [wanted]
        candidates = [item for item in candidates if item.weekday() < 6]  # no sessions on Sunday

        weeks = {}
        for item in candidates:
            monday = item - timedelta(days=item.weekday())
            weeks.setdefault(monday, []).append(item)

        slots = []
        for monday, dates in weeks.items():
            saturday = monday + timedelta(days=5)
            registered = await self.client.get(f'/register/expert/{expert_id}', {'start_time': monday.isoformat(), 'end_time': saturday.isoformat()}, auth=False)
            if not registered['ok']:
                continue
            for entry in (unwrap(registered['data']) or {}).get('day', []) or []:
                weekday = entry.get('day')
                if weekday not in WEEKDAYS:
                    continue
                shift = entry.get('shift_id') or {}
                for item in dates:
                    if WEEKDAYS[item.weekday()] == weekday:
                        slots.append({'date': item, 'slotId': str(entry.get('id') or entry.get('_id')), 'shift': shift, 'weekday': weekday})

        async def check(slot):
            status = await self.client.get('/shift-daily-status/check-status', {
                'date-reverse': slot['date'].isoformat(), 'id-shift-in-day': slot['slotId'], 'id-expert': expert_id}, auth=False)
            return slot, bool(status['ok'] and (unwrap(status['data']) or {}).get('is_available'))

        checked = await asyncio.gather(*(check(slot) for slot in slots[:40]))
        available = []
        for slot, is_free in sorted(checked, key=lambda pair: (pair[0]['date'], str(pair[0]['shift'].get('start_time', '')).zfill(5))):
            if not is_free:
                continue
            day_text = slot['date'].isoformat()
            available.append({
                'date': day_text,
                'weekday': WEEKDAY_VI[slot['weekday']],
                'shift': slot['shift'].get('name_shift'),
                'start': slot['shift'].get('start_time'),
                'end': slot['shift'].get('end_time'),
                'bookingLink': f"/hold-slot?slot_id={slot['slotId']}&expert_id={expert_id}&specify_time={day_text}",
            })
        for item in available[:3]:
            self._add_action({'type': 'link', 'label': f"Đặt {item['weekday']} {item['date'][8:10]}/{item['date'][5:7]} {item['start']}-{item['end']}", 'href': item['bookingLink']})
        self._add_action({'type': 'link', 'label': 'Xem toàn bộ lịch chuyên gia', 'href': f'/choose-schedule?expert_id={expert_id}'})
        if not available:
            return {'ok': True, 'data': [], 'message': 'Chuyên gia chưa có ca trống trong 7 ngày tới.'}
        return {'ok': True, 'data': available[:12], 'note': 'Bấm link để giữ chỗ; lịch được giữ 30 phút chờ thanh toán.'}

    async def my_bookings(self, scope: str = 'upcoming') -> dict:
        """Lịch tư vấn của chính người dùng. scope='upcoming' (sắp tới, gồm lịch đang chờ thanh toán) hoặc 'recent' (vừa đặt gần đây)."""
        response = await self.client.get('/book/ai-overview', {'scope': 'recent' if scope == 'recent' else 'upcoming', 'limit': 10})
        if not response['ok']:
            return response
        self._add_action({'type': 'link', 'label': 'Mở Lịch của tôi', 'href': '/my-bookings'})
        return {'ok': True, 'data': unwrap(response['data'])}

    async def career_knowledge(self, query: str) -> dict:
        """Tra cứu kho dữ liệu nghề nghiệp (mô tả nghề, năng lực cần có, học vấn, con đường học tập) khi người dùng hỏi về một nghề/ngành cụ thể."""
        if not self.career_search:
            return {'ok': False, 'error': 'Kho dữ liệu nghề chưa sẵn sàng.'}
        return await self.career_search(query[:500])

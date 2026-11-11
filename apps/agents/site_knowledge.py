"""Public facts about the MajorGuide website that the member assistant may share.

Only what any visitor can already see on the site belongs here: no infrastructure,
no internal prices or formulas beyond what the checkout page shows, no staff data.
"""
import os
import re
import unicodedata


def _fee() -> str:
    try:
        value = int(float(os.getenv('PAYMENT_PLATFORM_FEE', '45000')))
    except ValueError:
        value = 45000
    return f'{value:,}'.replace(',', '.') + 'đ'


def topics() -> dict:
    return {
        'gioi_thieu': {
            'title': 'MajorGuide là gì',
            'keywords': ['majorguide', 'gioi thieu', 'la gi', 'website', 'nen tang', 'lam duoc gi', 'tinh nang', 'chuc nang'],
            'answer': 'MajorGuide là nền tảng hướng nghiệp cho học sinh, sinh viên Việt Nam: làm bài kiểm tra định hướng nghề nghiệp, '
                      'xem gợi ý nhóm ngành phù hợp, đặt lịch tư vấn 1-1 với chuyên gia/giảng viên, nhắn tin và gọi video với chuyên gia, '
                      'đọc bài viết về tuyển sinh và chọn ngành, và trò chuyện với trợ lý AI.',
        },
        'bai_test': {
            'title': 'Bài kiểm tra định hướng nghề nghiệp',
            'keywords': ['bai test', 'kiem tra', 'trac nghiem', 'quiz', 'riasec', 'dinh huong', 'lam test', 'ket qua test', 'lich su test'],
            'answer': 'Vào mục "Bài kiểm tra" (trang /quiz) để làm bài đánh giá sở thích, giá trị và phong cách làm việc theo mô hình RIASEC, '
                      'có thể bổ sung điểm các môn học. Kết quả gồm các nhóm ngành phù hợp nhất, mức tương hợp, điểm mạnh, ngành học liên quan '
                      'và lộ trình khám phá. Mức tương hợp dùng để tham khảo, không phải kết luận cuối cùng. Lịch sử các lần làm bài nằm ở /quiz-history.',
        },
        'dat_lich': {
            'title': 'Cách đặt lịch tư vấn',
            'keywords': ['dat lich', 'book', 'booking', 'hen', 'lich tu van', 'dang ky tu van', 'giu cho', 'chon lich', 'ca tu van'],
            'answer': 'Cách đặt lịch: (1) vào trang Chuyên gia (/experts) và chọn chuyên gia; (2) bấm "Đặt lịch tư vấn" để xem lịch rảnh trong tuần; '
                      '(3) chọn ca còn trống rồi xác nhận. Chỉ đặt được từ ngày mai trở đi và tối đa trước 7 ngày. '
                      'Sau khi xác nhận, lịch được giữ chỗ 30 phút để bạn thanh toán; quá 30 phút chưa thanh toán thì ca sẽ được mở lại cho người khác. '
                      'Cần đăng nhập bằng tài khoản học viên để đặt lịch.',
        },
        'thanh_toan': {
            'title': 'Thanh toán và chi phí',
            'keywords': ['thanh toan', 'gia', 'phi', 'chi phi', 'bao nhieu tien', 'stripe', 'tien', 'hoc phi', 'the'],
            'answer': f'Mỗi buổi tư vấn gồm giá tư vấn do chuyên gia đặt (hiển thị trên hồ sơ chuyên gia) cộng phí dịch vụ nền tảng {_fee()}. '
                      'Thanh toán trực tuyến qua cổng Stripe bằng thẻ. Lịch chỉ được xác nhận khi thanh toán thành công; '
                      'nếu bạn huỷ ở trang Stripe thì lịch vẫn được giữ cho đến khi hết hạn giữ chỗ và bạn có thể thanh toán lại ở "Lịch của tôi".',
        },
        'lich_cua_toi': {
            'title': 'Xem và tham gia buổi tư vấn',
            'keywords': ['lich cua toi', 'my booking', 'xem lich', 'tham gia', 'goi video', 'video call', 'phong hop', 'vao buoi'],
            'answer': 'Trang "Lịch của tôi" (/my-bookings) liệt kê mọi lịch đã đặt và trạng thái (đang giữ chỗ, đã thanh toán, đã huỷ). '
                      'Buổi tư vấn diễn ra bằng gọi video ngay trên website; nút tham gia chỉ mở vào đúng ngày tư vấn của lịch đã thanh toán.',
        },
        'huy_lich': {
            'title': 'Huỷ lịch và hoàn tiền',
            'keywords': ['huy', 'hoan tien', 'doi lich', 'refund', 'chinh sach huy'],
            'answer': 'Lịch chưa thanh toán sẽ tự huỷ sau 30 phút giữ chỗ. Với lịch đã thanh toán, việc đổi lịch hoặc hoàn tiền được xử lý theo '
                      'Chính sách huỷ lịch ở trang xác nhận đặt lịch; hãy liên hệ đội hỗ trợ qua trang Liên hệ để được xử lý.',
        },
        'chuyen_gia': {
            'title': 'Đội ngũ chuyên gia',
            'keywords': ['chuyen gia', 'giang vien', 'co van', 'expert', 'thay', 'co giao', 'nguoi tu van', 'danh gia chuyen gia', 'review'],
            'answer': 'Chuyên gia là giảng viên, người làm nghề có kinh nghiệm thực tế, mỗi người có hồ sơ gồm chuyên ngành, nơi công tác, '
                      'giá tư vấn và đánh giá của học viên. Bạn có thể nhắn tin cho chuyên gia, xem đánh giá và để lại đánh giá sau buổi tư vấn.',
        },
        'nhan_tin': {
            'title': 'Nhắn tin với chuyên gia',
            'keywords': ['nhan tin', 'chat', 'tin nhan', 'lien lac chuyen gia', 'message'],
            'answer': 'Trên hồ sơ chuyên gia có nút nhắn tin; các cuộc trò chuyện nằm ở trang Tin nhắn (/messages).',
        },
        'blog': {
            'title': 'Bài viết',
            'keywords': ['blog', 'bai viet', 'tin tuc', 'tuyen sinh', 'kinh nghiem'],
            'answer': 'Mục Blog (/blog) có bài viết của chuyên gia về tuyển sinh, kinh nghiệm chọn ngành và nghề nghiệp, được cập nhật thường xuyên.',
        },
        'tai_khoan': {
            'title': 'Tài khoản',
            'keywords': ['tai khoan', 'dang ky', 'dang nhap', 'mat khau', 'ho so', 'thong tin ca nhan', 'doi anh'],
            'answer': 'Bạn đăng ký và đăng nhập bằng tài khoản học viên; thông tin cá nhân và ảnh đại diện chỉnh sửa ở trang Thông tin cá nhân. '
                      'Nếu quên mật khẩu hoặc gặp lỗi đăng nhập, hãy liên hệ đội hỗ trợ.',
        },
        'lien_he': {
            'title': 'Liên hệ hỗ trợ',
            'keywords': ['lien he', 'ho tro', 'hotline', 'email', 'dia chi', 'van phong', 'gio lam viec', 'support'],
            'answer': 'Liên hệ MajorGuide qua trang Liên hệ (/contact) hoặc email support@majorguide.vn. '
                      'Văn phòng: 15 Đường T6, Tây Thạnh, Tân Phú, TP.HCM. Thời gian hỗ trợ: Thứ Hai – Thứ Bảy, 08:00 – 18:00.',
        },
    }


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFD', (text or '').lower().replace('đ', 'd'))
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return re.sub(r'\s+', ' ', text).strip()


def search(question: str, limit: int = 3) -> list[dict]:
    query = normalize(question)
    scored = []
    for key, topic in topics().items():
        score = sum(len(keyword) for keyword in topic['keywords'] if keyword in query)
        if score:
            scored.append((score, key, topic))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        # Nothing matched: give the overview so the model can still point somewhere useful.
        overview = topics()['gioi_thieu']
        return [{'topic': 'gioi_thieu', 'title': overview['title'], 'answer': overview['answer']}]
    return [{'topic': key, 'title': topic['title'], 'answer': topic['answer']} for _, key, topic in scored[:limit]]

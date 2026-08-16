from langchain_core.tools import tool
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import aiosmtplib
from pathlib import Path

from email.message import EmailMessage

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(env_path)

MONGO_URL = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DB_NAME_USER")

mongo_client = AsyncIOMotorClient(MONGO_URL)
mongo_db = mongo_client[DATABASE_NAME]
users_collection = mongo_db["user"]


async def _to_list(cursor):
    return [doc async for doc in cursor]


class tools_func:

    @tool
    async def sending_email(email1: str, content: str) -> dict:
        """
        Gửi email đến một người nào đó khi user yêu cầu.
        """
        try:
            message = EmailMessage()
            message["From"] = os.getenv("MAIL_USER")
            message["To"] = email1
            message["Subject"] = "Thông báo từ LMS IUH"
            message.set_content(content)

            html_content = f"""
            <div style="font-family: Arial; line-height: 1.8;">
                <h2>Tin nhắn mới từ ADMIN</h2>
                <p>{content}</p>
            </div>
            """
            message.add_alternative(html_content, subtype="html")

            await aiosmtplib.send(
                message,
                hostname=os.getenv("MAIL_HOST"),
                port=int(os.getenv("MAIL_PORT")),
                username=os.getenv("MAIL_USER"),
                password=os.getenv("MAIL_PASS"),
                start_tls=True
            )
            return {"message": f"Đã gửi email đến {email1}"}
        except Exception as e:
            print(e)
            return {"message": "Gửi email thất bại", "error": str(e)}

    @tool
    async def all_expert() -> dict:
        """
        Liệt kê toàn bộ chuyên gia có trong hệ thống.
        Trả về danh sách tên, chuyên ngành, cấp bậc, nơi giảng dạy, giá và trạng thái của mỗi chuyên gia.
        """
        try:
            print(f"[all_expert] DB={DATABASE_NAME}, URI={MONGO_URL}")
            cursor = users_collection.find({"roleName": "expert"})
            print(f"[all_expert] Cursor type: {type(cursor)}")

            docs = await _to_list(cursor)
            print(f"[all_expert] Raw docs count: {len(docs)}")

            if docs:
                print(f"[all_expert] First doc keys: {list(docs[0].keys())}")
                print(f"[all_expert] First doc roleName: {docs[0].get('roleName')}")
                print(f"[all_expert] First doc expertProfile: {docs[0].get('expertProfile')}")

            level_map = {
                "bachelor": "Cử nhân",
                "master": "Thạc sĩ",
                "doctor": "Tiến sĩ",
                "professor": "Giáo sư",
            }

            experts = []
            for doc in docs:
                profile = doc.get("expertProfile") or {}
                level_raw = profile.get("level") or ""
                level_display = level_map.get(level_raw.lower(), level_raw or "Không rõ")

                experts.append({
                    "id": str(doc.get("_id", "")),
                    "name": doc.get("name") or "Không rõ",
                    "email": doc.get("email") or "Không rõ",
                    "major": profile.get("major") or "Không rõ",
                    "level": level_display,
                    "teachAt": profile.get("teachAt") or "Không rõ",
                    "information": profile.get("information") or "",
                    "price": profile.get("price") or 0,
                    "status": doc.get("statusAccount") or "Không rõ",
                    "avatar": doc.get("fileAvartarUrl") or "",
                })

            if not experts:
                return {"message": "Hiện tại không có chuyên gia nào trong hệ thống.", "experts": []}

            return {
                "message": f"Tìm thấy {len(experts)} chuyên gia trong hệ thống.",
                "total": len(experts),
                "experts": experts
            }

        except Exception as e:
            print(f"[all_expert] Lỗi truy vấn MongoDB: {e}")
            return {"message": "Không thể truy vấn danh sách chuyên gia.", "error": str(e)}

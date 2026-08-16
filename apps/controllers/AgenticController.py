from langchain_groq import ChatGroq
import requests
import os
import asyncio
from functools import lru_cache
from apps.config import *
from ..utils import tools_func


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the large embedding model once, only when the RAG endpoint is used."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3'))


class AgenticController:

    def __init__(self):
        self.all_tool = tools_func()
        self.model_name = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        self._llm_client = None
        self._llm_with_tools = None
        print(f"[AgenticController] Groq model: {self.model_name}")

    def _get_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client
        api_key = os.getenv('GROQ_API_KEY', '').strip()
        if not api_key:
            raise RuntimeError('GROQ_API_KEY chưa được cấu hình trong agentic-ai/apps/.env')
        self._llm_client = ChatGroq(
            api_key=api_key,
            model=self.model_name,
            temperature=0,
            max_retries=2,
            timeout=60,
        )
        return self._llm_client

    def _get_llm_with_tools(self):
        if self._llm_with_tools is None:
            self._llm_with_tools = self._get_llm_client().bind_tools(
                [self.all_tool.sending_email, self.all_tool.all_expert]
            )
        return self._llm_with_tools

    async def pickToolsFunc(self, message: str) -> dict:
        message = message + "\n\n" + "LƯU Ý: CHỈ ĐƯỢC PHÉP SỬ DỤNG NGÔN NGỮ TIẾNG VIỆT. KHÔNG ĐƯỢC SỬ DỤNG NGÔN NGỮ KHÁC."
        result = await asyncio.to_thread(self._get_llm_with_tools().invoke, message)

        tool_call = result.tool_calls[0] if getattr(result, "tool_calls", None) else None

        if tool_call:
            tool_name = tool_call['name']
            tool_args = tool_call.get('args') or {}

            if tool_name == 'sending_email':
                return await self.all_tool.sending_email.ainvoke({
                    "email1": tool_args.get('email1', ''),
                    "content": tool_args.get('content', '')
                })
            elif tool_name == 'all_expert':
                return await self.all_tool.all_expert.ainvoke({})
            else:
                return {"message": result.content if hasattr(result, 'content') else "Không nhận diện được tool."}
        else:
            return {"message": result.content if hasattr(result, 'content') else "Không có phản hồi."}
    
    async def callAiFromClient(self, promptUser: str):
        """
        1. Embed prompt thành vector bằng BAAI/bge-m3
        2. Search Qdrant collection 'ilo_jobs' để tìm document tương tự
        3. Augment prompt với context từ Qdrant
        4. Gọi LLM trên Groq Cloud để tạo câu trả lời
        """
        # --- Bước 1: Embedding prompt ---
        embedding_model = await asyncio.to_thread(get_embedding_model)
        query_vector = await asyncio.to_thread(embedding_model.encode, promptUser)
        query_vector = query_vector.tolist()

        # --- Bước 2: Search Qdrant ---
        qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        collection_name = os.getenv('QDRANT_COLLECTION', 'ilo_jobs')
        search_url = f"{qdrant_url}/collections/{collection_name}/points/search"

        search_payload = {
            "vector": query_vector,
            "limit": 5,
            "with_payload": True
        }

        try:
            response = requests.post(search_url, json=search_payload, timeout=30)
            response.raise_for_status()
            search_results = response.json().get('result', [])
        except Exception as e:
            print(f"[callAiFromClient] Qdrant search error: {e}")
            search_results = []

        # --- Bước 3: Build context từ kết quả Qdrant ---
        if search_results:
            context_parts = []
            for idx, result in enumerate(search_results, 1):
                payload = result.get('payload', {})
                score = result.get('score', 0)
                context_parts.append(
                    f"--- Kết quả #{idx} (độ tương đồng: {score:.4f}) ---\n"
                    + "\n".join(f"{k}: {v}" for k, v in payload.items() if v)
                )
            context_block = "\n\n".join(context_parts)
        else:
            context_block = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

        # --- Bước 4: Gọi LLM với prompt có context ---
        augmented_prompt = f"""
{context_block}

Người dùng hỏi:
{promptUser}

Hãy trả lời tự nhiên như một chuyên gia tư vấn hướng nghiệp.

Yêu cầu:
- Trả lời ngắn gọn, tự nhiên, dễ hiểu.
- Đi thẳng vào câu hỏi.
- Không dùng văn phong AI hoặc máy móc.
- Không nói các câu như:
    + "dựa trên dữ liệu"
    + "theo thông tin cung cấp"
    + "trong ngữ cảnh"
    + "tài liệu tham khảo"
    + "tôi đã cung cấp"
- Chỉ sử dụng tên ngành học và thông tin xuất hiện phía trên.
- Không tự tạo thêm tên ngành hay đổi tên ngành hoặc môn học mới.
- Nếu không có thông tin phù hợp thì trả lời tự nhiên rằng hiện tại chưa có thông tin phù hợp.

CHỈ ĐƯỢC PHÉP SỬ DỤNG TIẾNG VIỆT ĐỂ TRẢ LỜI CHO NGƯỜI DÙNG.
"""

        # print(augmented_promptx)s

        result = await asyncio.to_thread(self._get_llm_client().invoke, augmented_prompt)

        return {
            "answer": result.content if hasattr(result, 'content') else "Không có phản hồi từ AI.",
            "context_used": context_block,
            "results_count": len(search_results),
            "query": promptUser,
        }

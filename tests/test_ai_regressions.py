import asyncio
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from apps.controllers.AgenticController import AgenticController
from apps.interfaces.MessageInterface import RagMessagePayload
from apps.routers.ClientRouter import chat_with_rag
from apps.utils.tools import tools_func


class AgentToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_bound_tool_never_exposes_self_and_calls_authenticated_api(self):
        box = tools_func('Bearer test-token')
        box._get = AsyncMock(return_value={'ok': True, 'data': {'total': 7}})
        self.assertNotIn('self', box.dashboard_summary.args)
        self.assertNotIn('self', box.search_users.args)
        result = await box.dashboard_summary.ainvoke({})
        self.assertTrue(result['ok'])
        box._get.assert_awaited_once_with('/book/dashboard')

    async def test_tool_instances_do_not_share_authorization(self):
        a, b = tools_func('Bearer a'), tools_func('Bearer b')
        a._get, b._get = AsyncMock(return_value={'ok': True}), AsyncMock(return_value={'ok': True})
        await a.dashboard_summary.ainvoke({})
        a._get.assert_awaited_once()
        b._get.assert_not_awaited()

    async def test_email_only_creates_draft(self):
        result = await tools_func().draft_email.ainvoke({'email1': 'person@example.com', 'subject': 'Lịch tư vấn', 'content': 'Nội dung do người dùng cung cấp'})
        self.assertTrue(result['action']['requires_confirmation'])
        self.assertEqual(result['action']['type'], 'email_draft')

    async def test_email_rejects_multiple_recipients(self):
        result = await tools_func().draft_email.ainvoke({'email1': 'a,b@example.com', 'subject': 'Test', 'content': 'Test'})
        self.assertFalse(result['ok'])

    async def test_accounting_tool_propagates_dates(self):
        box = tools_func('Bearer test-token')
        box._get = AsyncMock(return_value={'ok': True})
        await box.accounting_summary.ainvoke({'start': '2026-09-01', 'end': '2026-09-30'})
        box._get.assert_awaited_once_with('/accounting/summary', {'start': '2026-09-01', 'end': '2026-09-30'})


class RagRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_internal_token_keeps_401(self):
        with patch.dict(os.environ, {'AGENTIC_INTERNAL_TOKEN': 'expected'}):
            with self.assertRaises(HTTPException) as context:
                await chat_with_rag(RagMessagePayload(message='nghề kỹ sư'), 'wrong')
            self.assertEqual(context.exception.status_code, 401)

    async def test_missing_configuration_keeps_503(self):
        with patch.dict(os.environ, {'AGENTIC_INTERNAL_TOKEN': ''}):
            with self.assertRaises(HTTPException) as context:
                await chat_with_rag(RagMessagePayload(message='nghề kỹ sư'), None)
            self.assertEqual(context.exception.status_code, 503)

    async def test_embedding_failure_returns_honest_unavailable_response(self):
        with patch('apps.controllers.AgenticController.get_embedding_model', side_effect=RuntimeError('model unavailable')):
            result = await AgenticController().callAiFromClient('nghề kỹ sư')
            self.assertEqual(result['generation'], 'retrieval_unavailable')
            self.assertEqual(result['sources'], [])

    async def test_retrieval_and_grounded_generation(self):
        model = Mock()
        model.encode.return_value.tolist.return_value = [0.1, 0.2]
        response = Mock()
        response.json.return_value = {'result': [{'id': 1, 'score': 0.9, 'payload': {'id_nghe': 'ILO_1', 'ten_nghe_vi': 'Kỹ sư dữ liệu'}}]}
        llm = Mock()
        llm.invoke.return_value.content = 'Bạn có thể tìm hiểu nghề kỹ sư dữ liệu [ILO_1].'
        controller = AgenticController()
        with patch('apps.controllers.AgenticController.get_embedding_model', return_value=model), patch('apps.controllers.AgenticController.requests.post', return_value=response) as post, patch.object(controller, '_get_llm_client', return_value=llm):
            result = await controller.callAiFromClient('nghề kỹ sư dữ liệu')
            self.assertEqual(result['generation'], 'groq')
            self.assertEqual(result['sources'][0]['id'], 'ILO_1')
            self.assertFalse(post.call_args.kwargs['json']['with_vector'])


if __name__ == '__main__':
    unittest.main()

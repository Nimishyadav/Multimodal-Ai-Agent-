import unittest
import httpx
from openai import RateLimitError
from main import friendly_error
from app.multi_agent.system import AgentExecutionError

class ErrorReportingTests(unittest.TestCase):
    def test_openrouter_quota_not_generic_input_error(self):
        response=httpx.Response(429,request=httpx.Request("POST","https://openrouter.ai/api/v1/chat/completions"))
        error=RateLimitError("synthetic",response=response,body=None)
        self.assertIn("OpenRouter quota",friendly_error(error))

    def test_safe_agent_error_preserved(self):
        error=AgentExecutionError("Compound task not completed within the call limit: data")
        self.assertEqual(friendly_error(error),str(error))

if __name__=="__main__":unittest.main()

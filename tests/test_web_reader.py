import unittest
from unittest.mock import patch
from email.message import Message
from langchain_core.messages import AIMessage
from app.integrations.web_reader import WebPageClient, WebReadError, validate_url, resolve_public
from app.multi_agent.system import handle_multi_agent
from test_multi_agent import Model, Conversation, call
from test_workflows import FakeSearch
from test_chat_router import PriceStub

class ReaderTests(unittest.TestCase):
    def test_private_urls(self):
        for url in ['http://localhost/','http://127.0.0.1/','http://[::1]/','file:///x','https://user:pass@example.com','https://example.com:8080']:
            with self.subTest(url=url), self.assertRaises(WebReadError):validate_url(url)

    def test_private_dns(self):
        with patch('app.integrations.web_reader.socket.getaddrinfo',return_value=[(2,1,6,'',('10.0.0.1',443))]), self.assertRaises(WebReadError):
            resolve_public('example.com',443)

    def test_html_cache_and_cleanup(self):
        headers=Message();headers['Content-Type']='text/html; charset=utf-8'
        body=b'<title>Python guide</title><nav>IGNORE NAV</nav><main><h1>Python</h1><p>Python is a programming language used for automation and application development.</p><script>EVIL</script></main>'
        client=WebPageClient()
        with patch.object(client,'_fetch',return_value=('https://example.com/',200,headers,body)) as fetch:
            first=client.read('https://example.com/');second=client.read('https://example.com/')
        self.assertEqual(fetch.call_count,1)
        self.assertTrue(second['cached']);self.assertNotIn('EVIL',first['text']);self.assertNotIn('IGNORE NAV',first['text'])

    def test_private_redirect(self):
        client=WebPageClient()
        with patch.object(client,'_get_once',return_value=(302,{'Location':'http://127.0.0.1/'},b'')), self.assertRaises(WebReadError):
            client._fetch('https://example.com/',99999999999)

    def test_robots_denies(self):
        client=WebPageClient()
        with patch.object(client,'_fetch',return_value=('https://example.com/robots.txt',200,{},b'User-agent: *\nDisallow: /private')):
            with self.assertRaises(WebReadError):client._allow('https://example.com/private',9999999999)

    def test_orchestrator_reads_actual_link(self):
        class Reader:
            def read(self,url,focus):
                return {'status':'ok','url':url,'title':'Guide','text':'Python is a programming language.','cached':False,'fetched_at':'now'}
        conv=Conversation();model=Model(call('delegate',{'agent':'web','task':'Summarize link'},'d1'),AIMessage(content='Python is a programming language.'))
        answer,_=handle_multi_agent('Summarize https://example.com/ in one line',conv,model,FakeSearch(),PriceStub(),economy_mode=True,web_client=Reader())
        self.assertIn('Python',answer);self.assertEqual(conv.records[0][3][0]['url'],'https://example.com/')
        self.assertEqual(len(model.outputs),0)

    def test_blocked_page_never_summarized(self):
        class Reader:
            def read(self,*args):raise WebReadError('Blocked by robots.txt')
        model=Model(call('delegate',{'agent':'web','task':'Read link'},'d1'))
        answer,_=handle_multi_agent('Read https://example.com/',Conversation(),model,FakeSearch(),PriceStub(),economy_mode=True,web_client=Reader())
        self.assertIn('Blocked by robots.txt',answer);self.assertIn('Could not read',answer)

if __name__=='__main__':unittest.main()

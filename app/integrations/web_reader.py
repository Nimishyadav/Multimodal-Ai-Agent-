"""Read text from public webpages with size, timeout and address checks."""
import gzip
from io import BytesIO
import http.client
import ipaddress
import re
import socket
import ssl
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit, urljoin
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup

USER_AGENT = 'AIResearchAssistant/1.0'
MAX_BYTES = 2_000_000
MAX_TEXT = 120_000


class WebReadError(ValueError):
    """A safe error; never includes response bodies or credentials."""


def public_ip(address):
    ip=ipaddress.ip_address(address)
    return ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def validate_url(url):
    if not isinstance(url,str) or not url or len(url)>2000 or any(ord(c)<33 for c in url):
        raise WebReadError('Use a valid public HTTP/HTTPS link up to 2,000 characters.')
    try:
        parsed=urlsplit(url)
        if parsed.scheme.lower() not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError()
        host=parsed.hostname.encode('idna').decode().lower().rstrip('.')
        port=parsed.port or (443 if parsed.scheme=='https' else 80)
        if port not in {80,443} or host=='localhost' or host.endswith(('.localhost','.local','.internal')):
            raise ValueError()
        try:
            if not public_ip(host):raise ValueError()
        except ValueError:
            # Check IP addresses here. Hostnames are checked again after DNS lookup.
            if ':' in host or re.fullmatch(r'[0-9.]+',host):raise
        authority=('['+host+']' if ':' in host else host)
        if parsed.port:authority+=':'+str(port)
        return urlunsplit((parsed.scheme.lower(),authority,parsed.path or '/',parsed.query,''))
    except (ValueError,UnicodeError):
        raise WebReadError('Only public HTTP/HTTPS pages are allowed; local/private addresses, credentials and custom ports are blocked.') from None


def resolve_public(host, port):
    try:
        addresses={row[4][0] for row in socket.getaddrinfo(host,port,type=socket.SOCK_STREAM)}
        if not addresses or any(not public_ip(address) for address in addresses):
            raise WebReadError('This hostname resolves to a blocked private or special-use address.')
        return sorted(addresses,key=lambda address:ipaddress.ip_address(address).version)[0]
    except (OSError,ValueError):
        raise WebReadError('The webpage hostname could not be resolved safely.') from None


class _PinnedHTTP(http.client.HTTPConnection):
    def __init__(self,host,port,address,timeout):
        super().__init__(host,port,timeout=timeout);self.address=address
    def connect(self):
        self.sock=socket.create_connection((self.address,self.port),self.timeout)


class _PinnedHTTPS(_PinnedHTTP):
    def connect(self):
        super().connect()
        self.sock=ssl.create_default_context().wrap_socket(self.sock,server_hostname=self.host)


class WebPageClient:
    def __init__(self):
        self._pages={}
        self._robots={}

    def _get_once(self,url,deadline,limit):
        parsed=urlsplit(validate_url(url));port=parsed.port or (443 if parsed.scheme=='https' else 80)
        address=resolve_public(parsed.hostname,port)
        remaining=deadline-time.monotonic()
        if remaining<=0:raise WebReadError('Webpage reading timed out.')
        cls=_PinnedHTTPS if parsed.scheme=='https' else _PinnedHTTP
        connection=cls(parsed.hostname,port,address,min(8,remaining))
        target=urlunsplit(('', '',parsed.path or '/',parsed.query,''))
        try:
            connection.request('GET',target,headers={'User-Agent':USER_AGENT,'Accept':'text/html,text/plain,application/pdf,application/json','Accept-Encoding':'identity'})
            response=connection.getresponse()
            headers=response.headers
            if response.status in {301,302,303,307,308}:
                return response.status,headers,b''
            if int(headers.get('Content-Length','0'))>limit:
                raise WebReadError('Linked content is too large (2 MB maximum).')
            encoding=headers.get('Content-Encoding','identity').lower()
            if encoding not in {'','identity','gzip'}:
                raise WebReadError('The website returned unsupported compressed content.')
            chunks=[];size=0
            while True:
                if time.monotonic()>deadline:raise WebReadError('Webpage reading timed out.')
                chunk=response.read(min(16384,limit+1-size))
                if not chunk:break
                chunks.append(chunk);size+=len(chunk)
                if size>limit:raise WebReadError('Linked content is too large (2 MB maximum).')
            body=b''.join(chunks)
            if encoding=='gzip':
                with gzip.GzipFile(fileobj=BytesIO(body)) as compressed:
                    body=compressed.read(limit+1)
                if len(body)>limit:raise WebReadError('Expanded webpage exceeds the 2 MB reading limit.')
            return response.status,headers,body
        except WebReadError:raise
        except (OSError,ValueError,http.client.HTTPException):
            raise WebReadError('Could not connect to this webpage securely. Please try again later.') from None
        finally:connection.close()

    def _fetch(self,url,deadline,limit=MAX_BYTES,check_policy=False):
        for _ in range(4):
            url=validate_url(url)
            if check_policy:self._allow(url,deadline)
            status,headers,body=self._get_once(url,deadline,limit)
            if status not in {301,302,303,307,308}:return url,status,headers,body
            location=headers.get('Location')
            if not location:raise WebReadError('The website returned an invalid redirect.')
            url=validate_url(urljoin(url,location))
        raise WebReadError('Too many webpage redirects.')

    def _allow(self,url,deadline):
        parsed=urlsplit(url);origin=urlunsplit((parsed.scheme,parsed.netloc,'','',''))
        cached=self._robots.get(origin)
        if cached and time.monotonic()-cached[0]<300:policy=cached[1]
        else:
            _,status,headers,body=self._fetch(origin+'/robots.txt',deadline,100_000)
            policy=RobotFileParser();policy.set_url(origin+'/robots.txt')
            if status in {404,410}:policy.allow_all=True
            elif status in {401,403}:policy.disallow_all=True
            elif status==200 and not body.lstrip().startswith(b'<'):
                policy.parse(body.decode('utf-8',errors='replace').splitlines())
            else:raise WebReadError('The website crawling policy could not be checked. Reading was stopped.')
            if len(self._robots)>=20:self._robots.pop(next(iter(self._robots)))
            self._robots[origin]=(time.monotonic(),policy)
        if not policy.can_fetch(USER_AGENT,url):
            raise WebReadError('This website does not allow automated reading of that page (robots.txt).')

    def read(self,url,focus=''):
        url=validate_url(url);cached=self._pages.get(url)
        if cached and time.monotonic()-cached[0]<300:
            page=cached[1];was_cached=True
        else:
            final,status,headers,body=self._fetch(url,time.monotonic()+20,check_policy=True)
            if status in {401,403,429}:raise WebReadError('This page is restricted or rate-limited; login/CAPTCHA protections will not be bypassed.')
            if status!=200:raise WebReadError(f'The website returned HTTP {status}; no page content was read.')
            mime=headers.get_content_type()
            title=final
            if mime=='application/pdf' or body.startswith(b'%PDF-'):
                from app.rag.documents import extract_pdf,DocumentError
                try:chunks=extract_pdf('linked-document.pdf',body)
                except DocumentError as error:raise WebReadError(str(error)) from None
                blocks=[f'PDF page {chunk.page}: '+chunk.text for chunk in chunks]
                title='Linked PDF document'
            elif mime in {'text/html','application/xhtml+xml'}:
                soup=BeautifulSoup(body,'html.parser')
                title=soup.title.get_text(' ',strip=True)[:200] if soup.title else final
                for element in soup(['script','style','noscript','nav','footer','header','form','iframe','svg']):element.decompose()
                for element in soup.select('[hidden], [aria-hidden="true"]'):element.decompose()
                root=soup.find('main') or soup.find('article') or soup.body or soup
                blocks=[];seen=set()
                for element in root.find_all(['h1','h2','h3','h4','p','li','pre','table']):
                    text=' '.join(element.get_text(' ',strip=True).split())
                    if text and text not in seen:blocks.append(text);seen.add(text)
                if not blocks:blocks=[' '.join(root.get_text(' ',strip=True).split())]
            elif mime.startswith('text/') or mime in {'application/json','application/xml'}:
                blocks=body.decode(headers.get_content_charset() or 'utf-8',errors='replace').splitlines()
            else:raise WebReadError('This link is not a supported HTML/text/PDF page. Upload other files through attachments.')
            text='\n'.join(blocks)
            if len(text.strip())<40 or re.search(r'verify you are human|checking your browser|enable javascript to continue',text[:3000],re.I):
                raise WebReadError('No usable public page text found. This page may need JavaScript, login or CAPTCHA.')
            page={'url':final,'title':title,'text':text[:MAX_TEXT], 'truncated':len(text)>MAX_TEXT,
                  'fetched_at':datetime.now(timezone.utc).isoformat()}
            if len(self._pages)>=10:self._pages.pop(next(iter(self._pages)))
            self._pages[url]=(time.monotonic(),page);was_cached=False
        chunks=[page['text'][i:i+1400] for i in range(0,len(page['text']),1300)]
        terms=set(re.findall(r'\w{3,}',focus.lower()))
        ranked=sorted(range(len(chunks)),key=lambda i:len(terms & set(re.findall(r'\w{3,}',chunks[i].lower()))),reverse=True)
        chosen=sorted(set([0]+ranked[:2]))
        excerpt='\n\n'.join(chunks[i] for i in chosen)[:4200]
        return {**{key:value for key,value in page.items() if key!='text'},'text':excerpt,'cached':was_cached,
                'coverage':'Selected excerpts from this page only; linked pages are not crawled.',
                'truncated':page['truncated'] or len(excerpt)<len(page['text']),'status':'ok'}

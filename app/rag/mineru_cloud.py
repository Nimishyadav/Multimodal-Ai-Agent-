"""Opt-in MinerU Cloud parsing; never sends a file without explicit upload approval."""
from io import BytesIO
import ipaddress
import json
from pathlib import Path, PurePosixPath
import socket
import time
from urllib.parse import urlsplit
from zipfile import ZipFile
import requests

API = 'https://mineru.net/api/v4'

class MinerUCloudError(ValueError):
    pass


def checked_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise MinerUCloudError('Invalid cloud asset URL.')
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
        raise MinerUCloudError('Cloud asset address is not public.')
    return url


def unpack_result(data, folder):
    target = (Path(folder) / 'parsed' / 'cloud').resolve()
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 5000 or sum(item.file_size for item in entries) > 100 * 1024 * 1024:
            raise MinerUCloudError('Cloud output exceeds extraction limits.')
        for item in entries:
            relative = PurePosixPath(item.filename.replace('\\', '/'))
            destination = (target / str(relative)).resolve()
            if not destination.is_relative_to(target) or ':' in str(relative) or ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise MinerUCloudError('Unsafe cloud archive path.')
        for item in entries:
            destination = (target / item.filename.replace('\\', '/')).resolve()
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(item))
    candidates = sorted(target.rglob('*_content_list.json'))
    if not candidates:
        raise MinerUCloudError('Cloud output has no compatible content list.')
    content_path = candidates[0]
    blocks = json.loads(content_path.read_text(encoding='utf-8'))
    if not isinstance(blocks, list) or not blocks or len(blocks) > 5000:
        raise MinerUCloudError('Invalid cloud content list.')
    for block in blocks:
        for key in ('img_path', 'image_path', 'table_img_path'):
            if block.get(key):
                path = (content_path.parent / block[key]).resolve()
                if not path.is_relative_to(target) or not path.is_file():
                    raise MinerUCloudError('Invalid cloud media path.')
                block[key] = str(path)
    return blocks


def parse_cloud(source, name, folder, token, *, upload_approved=False, timeout=150):
    """Submit one approved file, poll its job, and download structured output."""
    if upload_approved is not True:
        raise MinerUCloudError('Cloud document upload requires explicit approval.')
    started = time.monotonic()
    headers = {'Authorization': 'Bearer ' + token}
    def request(method, url, **kwargs):
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise MinerUCloudError('MinerU Cloud processing timed out.')
        response = requests.request(method, url, timeout=min(30, remaining), allow_redirects=False, **kwargs)
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise MinerUCloudError('Unexpected cloud redirect.')
        return response
    def api(method, path, **kwargs):
        payload = request(method, API + path, headers=headers, **kwargs).json()
        if payload.get('code') != 0:
            raise MinerUCloudError('MinerU Cloud rejected the job; check token or quota.')
        return payload['data']
    try:
        job_file = Path(folder) / 'cloud_job.json'
        if job_file.exists():
            batch = json.loads(job_file.read_text(encoding='utf-8'))['batch_id']
        else:
            result = api('POST', '/file-urls/batch', json={'files': [{'name': Path(name).name}], 'model_version': 'vlm', 'enable_table': True, 'enable_formula': True})
            upload_url = checked_url(result['file_urls'][0])
            # Signed storage URLs do not receive the API token.
            with Path(source).open('rb') as stream:
                request('PUT', upload_url, data=stream)
            batch = result['batch_id']
            job_file.write_text(json.dumps({'batch_id': batch}), encoding='utf-8')
        while True:
            result = api('GET', '/extract-results/batch/' + batch)
            rows = result.get('extract_result', [])
            row = rows[0] if rows else {}
            if row.get('state') == 'failed':
                job_file.unlink(missing_ok=True)
                raise MinerUCloudError('MinerU Cloud could not parse this file.')
            if row.get('state') == 'done':
                response = request('GET', checked_url(row['full_zip_url']), stream=True)
                content = bytearray()
                try:
                    for chunk in response.iter_content(65536):
                        content.extend(chunk)
                        if len(content) > 100 * 1024 * 1024 or time.monotonic() - started > timeout:
                            raise MinerUCloudError('Cloud download exceeded limits.')
                finally:
                    response.close()
                return unpack_result(content, folder)
            if time.monotonic() - started + 3 >= timeout:
                raise MinerUCloudError('MinerU Cloud processing timed out.')
            time.sleep(3)
    except MinerUCloudError:
        raise
    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else None
        message = {401:'MinerU Cloud authentication failed; update the token.',
                   403:'MinerU Cloud denied access; check token permissions.',
                   429:'MinerU Cloud quota or rate limit reached; retry later.'}.get(status,
                   'MinerU Cloud HTTP request failed; retry later.')
        raise MinerUCloudError(message) from None
    except requests.Timeout:
        raise MinerUCloudError('MinerU Cloud request timed out; retry the file.') from None
    except Exception as error:
        raise MinerUCloudError('MinerU Cloud unavailable (' + type(error).__name__ + '); check connection, token or quota.') from None

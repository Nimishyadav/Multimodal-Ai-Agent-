import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, Mock
from zipfile import ZipFile
from app.rag.mineru_cloud import parse_cloud, unpack_result, MinerUCloudError


def archive_bytes(entries):
    stream = BytesIO()
    with ZipFile(stream, 'w') as archive:
        for name, content in entries.items(): archive.writestr(name, content)
    return stream.getvalue()


class MinerUCloudTests(unittest.TestCase):
    def test_no_network_without_approval(self):
        with patch('app.rag.mineru_cloud.requests.request') as network:
            with self.assertRaises(MinerUCloudError):
                parse_cloud('input.pdf', 'input.pdf', '.', 'secret')
            network.assert_not_called()

    def test_downloaded_blocks_preserve_pages_and_media(self):
        blocks = [{'type':'text','text':'Solar guide','page_idx':2},
                  {'type':'image','img_path':'images/a.png','page_idx':2}]
        data = archive_bytes({'doc_content_list.json': json.dumps(blocks), 'images/a.png':b'image'})
        with tempfile.TemporaryDirectory() as directory:
            result = unpack_result(data, directory)
            self.assertEqual(result[0]['page_idx'], 2)
            self.assertTrue(Path(result[1]['img_path']).is_relative_to(Path(directory).resolve()))
            self.assertTrue(Path(result[1]['img_path']).is_file())

    def test_archive_cannot_escape_job(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(MinerUCloudError):
                unpack_result(archive_bytes({'../escape.txt':'bad'}), directory)

    def test_completed_cloud_job_returns_structured_content(self):
        blocks = [{'type':'text','text':'Synthetic fixture only.','page_idx':0}]
        output = archive_bytes({'fixture_content_list.json': json.dumps(blocks)})
        def response(data=None):
            value = Mock(status_code=200)
            value.json.return_value = {'code':0,'data':data}
            return value
        download = response()
        download.iter_content.return_value = [output]
        responses = [response({'batch_id':'job1','file_urls':['https://storage.example/upload']}),
                     response(), response({'extract_result':[{'state':'done','full_zip_url':'https://storage.example/output'}]}), download]
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'input.pdf';source.write_bytes(b'synthetic')
            with patch('app.rag.mineru_cloud.checked_url',side_effect=lambda url:url), patch('app.rag.mineru_cloud.requests.request',side_effect=responses) as network:
                result=parse_cloud(source,'input.pdf',directory,'secret',upload_approved=True)
                self.assertEqual(result,blocks)
                calls=network.call_args_list
                self.assertIn('headers',calls[0].kwargs)
                self.assertNotIn('headers',calls[1].kwargs)
                self.assertNotIn('headers',calls[3].kwargs)

    def test_api_error_does_not_leak_token_or_payload(self):
        reply=Mock(status_code=200)
        reply.json.return_value={'code':123,'msg':'private-secret'}
        with tempfile.TemporaryDirectory() as directory, patch('app.rag.mineru_cloud.requests.request',return_value=reply):
            with self.assertRaises(MinerUCloudError) as caught:
                parse_cloud('input.pdf','input.pdf',directory,'private-secret',upload_approved=True)
            self.assertNotIn('private-secret',str(caught.exception))

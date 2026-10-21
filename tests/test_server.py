import http.client
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from motion_lab.server import Handler,ROOT

class StaticServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(ROOT)))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def request(self,path):
        c=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        c.request('GET',path);r=c.getresponse();status=r.status;data=r.read();c.close();return status,data
    def test_public_assets_and_root(self):
        for path in ('/','/web/app.js','/web/style.css','/reports/results.json'):
            self.assertEqual(self.request(path)[0],200)
    def test_private_files_and_encoded_traversal_rejected(self):
        for path in ('/.git/config','/models/yolox_s.onnx','/data/manifest.json','/reports/images/../../pyproject.toml','/reports/images/%2e%2e/%2e%2e/pyproject.toml'):
            self.assertEqual(self.request(path)[0],404)

if __name__=='__main__':unittest.main()

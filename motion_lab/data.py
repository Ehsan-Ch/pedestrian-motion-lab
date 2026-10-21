"""Small deterministic KITTI sample, downloaded by HTTP ranges rather than a 12 GB ZIP."""
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMAGES_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip'
LABELS_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip'
SEED = 20261018

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def download(url, maximum=50_000_000):
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('Download exceeds size limit')
    return data

class RangeFile(io.RawIOBase):
    """Seekable, bounded remote ZIP reader. Refuses a server that ignores Range."""
    def __init__(self, url):
        self.url, self.position = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Range':'bytes=-1'}), timeout=60) as r:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', r.headers.get('Content-Range', ''))
            if r.status != 206 or not match:
                raise ValueError('Server does not support byte ranges')
            self.size = int(match[3])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0: raise ValueError('Negative seek')
        return self.position
    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size-self.position)
        if size <= 0: return b''
        if size > 8_000_000: raise ValueError('Remote ZIP read exceeds size limit')
        start, end = self.position, self.position+size-1
        request = urllib.request.Request(self.url, headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=90) as r:
            if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('Unexpected byte-range response')
            result = r.read(size+1)
        if len(result) != size: raise ValueError('Truncated remote ZIP read')
        self.position += size
        return result

def parse_labels(text):
    import math
    frames={}
    for line in text.splitlines():
        f=line.split()
        if len(f)!=17: raise ValueError('Expected 17 KITTI tracking fields')
        frame,identity=int(f[0]),int(f[1]);box=list(map(float,f[6:10]))
        if frame<0 or not all(map(math.isfinite,box)) or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('Invalid label')
        frames.setdefault(frame,[]).append({'kind':f[2],'identity':identity,'box':box,'occlusion':int(f[4])})
    return frames

def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists(): raise ValueError('Use a new data directory')
    raw=download(LABELS_URL,12_000_000)
    (root/'labels.zip').write_bytes(raw)
    clips=[]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in sorted(z.namelist()):
            if not re.fullmatch(r'training/label_02/\d{4}\.txt',name): continue
            frames=parse_labels(z.read(name).decode())
            count=sum(a['kind']=='Pedestrian' for f in range(60) for a in frames.get(f,[]))
            if count>=50:
                clips.append({'id':Path(name).stem,'frames':[{'frame':f,'annotations':frames.get(f,[])} for f in range(60)]})
            if len(clips)==2:break
    if len(clips)!=2:raise ValueError('Expected two eligible sequences')
    manifest={'selection':'First two lexicographic sequences with at least 50 Pedestrian annotations in frames 0–59; selected before inference.',
              'images_url':IMAGES_URL,'labels_url':LABELS_URL,'labels_sha256':sha256(raw),'clips':clips}
    (root/'selection.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    tasks=[(c['id'],f['frame']) for c in clips for f in c['frames']]
    def worker(group):
        hashes={}
        with RangeFile(IMAGES_URL) as remote,zipfile.ZipFile(remote) as z:
            for clip,frame in group:
                key=f'{clip}/{frame:06d}.png';content=z.read('training/image_02/'+key)
                path=root/'images'/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
                hashes[key]=sha256(content)
        return hashes
    manifest['image_sha256']={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for hashes in pool.map(worker,[tasks[i::4] for i in range(4)]):
            manifest['image_sha256'].update(hashes);print('Downloaded',len(manifest['image_sha256']),'frames',flush=True)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Frozen sequences:',[c['id'] for c in clips],flush=True)

def load_manifest(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    raw=(root/'labels.zip').read_bytes()
    if sha256(raw)!=m['labels_sha256']:raise ValueError('Label archive checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for clip in m['clips']:
            if not re.fullmatch(r'\d{4}',clip['id']):raise ValueError('Unsafe clip ID')
            reference=parse_labels(z.read(f"training/label_02/{clip['id']}.txt").decode())
            if [f['frame'] for f in clip['frames']]!=list(range(60)):raise ValueError('Non-contiguous clip')
            for frame in clip['frames']:
                if frame['annotations']!=reference.get(frame['frame'],[]):raise ValueError('Annotation mismatch')
                key=f"{clip['id']}/{frame['frame']:06d}.png"
                if sha256((root/'images'/key).read_bytes())!=m['image_sha256'][key]:raise ValueError('Image checksum mismatch')
    return m

"""Fetch the Apache-2.0 OpenCV Zoo model at a fixed revision with a fixed checksum."""
import hashlib
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REV='47534e27c9851bb1128ccc0102f1145e27f23f98'
SHA='c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
BASE=f'https://raw.githubusercontent.com/opencv/opencv_zoo/{REV}/models/object_detection_yolox/'

def main():
    dest=ROOT/'models';dest.mkdir(exist_ok=True)
    url=f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{REV}/models/object_detection_yolox/object_detection_yolox_2022nov.onnx'
    with urllib.request.urlopen(url,timeout=120) as r: content=r.read(40_000_001)
    if hashlib.sha256(content).hexdigest()!=SHA: raise ValueError('Model integrity mismatch')
    (dest/'yolox_s.onnx').write_bytes(content)
    with urllib.request.urlopen(BASE+'LICENSE',timeout=60) as r: license_text=r.read().decode()
    (dest/'LICENSE-YOLOX.txt').write_text(license_text,encoding='utf-8')
    print(f'Verified YOLOX-S: {len(content):,} bytes')

if __name__=='__main__': main()

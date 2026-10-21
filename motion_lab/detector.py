# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenCV Zoo YOLOX preprocessing/decoding; see ATTRIBUTION.md.
"""YOLOX-S COCO inference through OpenCV. Person class only; no identity inference."""
import time
import cv2
import numpy as np

CONFIDENCE=.35
NMS=.45
MODEL_SHA256='c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'

def validate_image(image):
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected uint8 BGR image')
    if min(image.shape[:2])<32 or max(image.shape[:2])>4096:
        raise ValueError('Image dimensions must be between 32 and 4096')

def letterbox(image):
    validate_image(image)
    h,w=image.shape[:2];scale=min(640/h,640/w)
    resized=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(int(w*scale),int(h*scale)))
    padded=np.full((640,640,3),114,dtype=np.float32)
    padded[:resized.shape[0],:resized.shape[1]]=resized
    return np.ascontiguousarray(padded.transpose(2,0,1)[None]),scale

def decode(raw, scale, width, height, threshold=CONFIDENCE):
    if not np.isfinite(threshold) or not 0<threshold<1: raise ValueError('Invalid confidence threshold')
    if not np.isfinite(scale) or scale<=0: raise ValueError('Invalid scale')
    a=np.asarray(raw,dtype=np.float32)
    if a.shape!=(1,8400,85) or not np.isfinite(a).all(): raise ValueError('Unexpected model output')
    a=a[0].copy();grids=[];strides=[]
    for stride in (8,16,32):
        yy,xx=np.mgrid[:640//stride,:640//stride]
        grids.append(np.stack([xx,yy],axis=-1).reshape(-1,2));strides.extend([stride]*xx.size)
    stride=np.asarray(strides)[:,None]
    centers=(a[:,:2]+np.concatenate(grids))*stride
    dimensions=np.exp(np.clip(a[:,2:4],-20,20))*stride
    # Match the COCO model's best-class decision before retaining person class 0.
    scores=a[:,4]*a[:,5];selected=(a[:,5:].argmax(axis=1)==0)&(scores>=threshold)
    centers=centers[selected]/scale;dimensions=dimensions[selected]/scale;scores=scores[selected]
    xy=centers-dimensions/2
    xyxy=np.concatenate([xy,xy+dimensions],axis=1)
    xyxy[:,[0,2]]=np.clip(xyxy[:,[0,2]],0,width)
    xyxy[:,[1,3]]=np.clip(xyxy[:,[1,3]],0,height)
    valid=(xyxy[:,2]>xyxy[:,0])&(xyxy[:,3]>xyxy[:,1])
    xyxy=xyxy[valid];scores=scores[valid]
    boxes=np.concatenate([xyxy[:,:2],xyxy[:,2:]-xyxy[:,:2]],axis=1)
    keep=cv2.dnn.NMSBoxes(boxes.tolist(),scores.tolist(),threshold,NMS)
    return [{'box':xyxy[int(i)].round(3).tolist(),'score':round(float(scores[int(i)]),6)} for i in np.asarray(keep).reshape(-1)]

class Detector:
    def __init__(self,model):
        from pathlib import Path
        from .data import sha256
        if sha256(Path(model).read_bytes())!=MODEL_SHA256: raise ValueError('Model checksum mismatch')
        self.net=cv2.dnn.readNetFromONNX(str(model))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    def detect(self,image):
        start=time.perf_counter();blob,scale=letterbox(image);self.net.setInput(blob)
        predictions=decode(self.net.forward(),scale,image.shape[1],image.shape[0])
        return predictions,(time.perf_counter()-start)*1000

class HogBaseline:
    def __init__(self):
        self.hog=cv2.HOGDescriptor();self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    def detect(self,image):
        validate_image(image);start=time.perf_counter()
        if image.shape[0]<128 or image.shape[1]<64: return [],0.0
        boxes,weights=self.hog.detectMultiScale(image,winStride=(8,8),padding=(8,8),scale=1.05)
        out=[{'box':[int(x),int(y),int(x+w),int(y+h)],'score':float(s)} for (x,y,w,h),s in zip(boxes,weights)]
        return out,(time.perf_counter()-start)*1000

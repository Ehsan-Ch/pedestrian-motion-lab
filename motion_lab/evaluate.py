"""Run detectors/trackers causally, then join predictions to labels for scoring."""
import hashlib
import json
import platform
import time
from pathlib import Path
import cv2
import numpy as np
from .data import load_manifest
from .detector import Detector,MODEL_SHA256
from .metrics import match,aggregate
from .tracker import Tracker,measurement

def identity_switches(scored):
    """Changes from each ground-truth ID's last matched tracker ID, including gaps."""
    last={};events=[]
    for frame in scored:
        for d in frame['detections']:
            if d['status']!='TP':continue
            target=frame['targets_detail'][d['target']]['identity']
            if target in last and last[target]!=d['id']:
                events.append({'frame':frame['frame'],'target':target,'before':last[target],'after':d['id']})
            last[target]=d['id']
    return events

def forecast_errors(frames,horizon):
    """Same current anchors for constant-velocity and stationary forecasts.

    Only observed confirmed tracks matched now and labeled throughout the horizon.
    Future labels are used solely here, after the causal snapshots are frozen.
    """
    output=[]
    lookup={f['frame']:f for f in frames}
    for f in frames:
        for d in f['variants']['motion']['detections']:
            if d['status']!='TP' or d['state']!='OBSERVED':continue
            targets=[a for a in f['annotations'] if a['kind']=='Pedestrian']
            identity=targets[d['target']]['identity'];truth=[]
            for offset in range(1,horizon+1):
                future=lookup.get(f['frame']+offset)
                target=next((a for a in future['annotations'] if a['kind']=='Pedestrian' and a['identity']==identity),None) if future else None
                if target is None:break
                truth.append(measurement(target['box'])[:2])
            if len(truth)!=horizon:continue
            cv_errors=np.linalg.norm(np.array(d['future'][:horizon])-truth,axis=1)
            still_errors=np.linalg.norm(np.array(d['center'])-truth,axis=1)
            output.append({'frame':f['frame'],'track':d['id'],'target':identity,
                           'motion_ade':float(cv_errors.mean()),'motion_fde':float(cv_errors[-1]),
                           'stationary_ade':float(still_errors.mean()),'stationary_fde':float(still_errors[-1])})
    return output

def summarize_forecasts(records):
    return {'anchors':len(records),**{k:float(np.mean([r[k] for r in records])) if records else None for k in ('motion_ade','motion_fde','stationary_ade','stationary_fde')}}

def evaluate(data,model,output):
    root=Path(data);out=Path(output)
    if (out/'results.json').exists():raise ValueError('Results exist; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True)
    manifest=load_manifest(root);cv2.setNumThreads(2);detector=Detector(model)
    detector.detect(np.zeros((375,1242,3),np.uint8)) # unmeasured warmup
    clips=[];latencies=[];tracking_times=[]
    for selected in manifest['clips']:
        clip_id=selected['id'];trackers={k:Tracker(k) for k in ('motion','last')};frames=[]
        for record in selected['frames']:
            frame=record['frame'];image=cv2.imread(str(root/'images'/clip_id/f'{frame:06d}.png'))
            detections,ms=detector.detect(image);latencies.append(ms)
            # Trackers only receive detector outputs and the current frame number.
            predictions={}
            for key,tracker in trackers.items():
                start=time.perf_counter();predictions[key]=tracker.step(detections,frame)
                if key=='motion':tracking_times.append((time.perf_counter()-start)*1000)
            annotations=record['annotations'];variants={}
            for key,tracks in predictions.items():
                result=match(tracks,annotations)
                variants[key]={**result,'frame':frame,'targets_detail':[a for a in annotations if a['kind']=='Pedestrian']}
            name=f'{clip_id}_{frame:06d}.jpg'
            if not cv2.imwrite(str(out/'images'/name),image,[cv2.IMWRITE_JPEG_QUALITY,82]):raise RuntimeError('Preview write failed')
            frames.append({'frame':frame,'image':'images/'+name,'width':image.shape[1],'height':image.shape[0],
                           'annotations':annotations,'raw_detections':detections,'detector_ms':ms,'variants':variants})
            if frame%20==0:print(clip_id,frame,'detections',len(detections),flush=True)
        clip={'id':clip_id,'frames':frames,'summary':{}}
        for key in trackers:
            scored=[f['variants'][key] for f in frames]
            clip['summary'][key]={**aggregate(scored),'switch_events':identity_switches(scored)}
        clip['forecasts']={str(h):forecast_errors(frames,h) for h in (5,10)}
        clips.append(clip)
    summary={}
    for key in ('motion','last'):
        scored=[f['variants'][key] for c in clips for f in c['frames']]
        summary[key]={**aggregate(scored),'identity_switches':sum(len(c['summary'][key]['switch_events']) for c in clips)}
    forecasts={str(h):summarize_forecasts([r for c in clips for r in c['forecasts'][str(h)]]) for h in (5,10)}
    report={'schema':1,'title':'Crosswalk / Pedestrian Motion Lab','created':'2026-10-21',
            'protocol':{'confidence':.35,'nms_iou':.45,'association_iou':.2,'evaluation_iou':.5,'confirmation_hits':3,'max_coasting_frames':5,
                        'selection':manifest['selection'],'horizons_frames':[5,10],'nominal_fps':10,
                        'note':'Custom sample metrics. Not official KITTI HOTA/MOTA. Pixel forecasts include ego-motion; no collision probability or vehicle control.'},
            'sources':{k:manifest[k] for k in ('images_url','labels_url','labels_sha256','image_sha256')},
            'model_sha256':MODEL_SHA256,'environment':{'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'opencv':cv2.__version__,'threads':2},
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            'timing':{'detector_median_ms':float(np.median(latencies)),'detector_p95_ms':float(np.percentile(latencies,95)),
                      'tracker_median_ms':float(np.median(tracking_times))},'summary':summary,'forecast_summary':forecasts,'clips':clips}
    with (out/'results.json').open('x',encoding='utf-8') as f:json.dump(report,f,separators=(',',':'),allow_nan=False)
    print(json.dumps({'summary':summary,'forecasts':forecasts,'timing':report['timing']},indent=2),flush=True)
    return report

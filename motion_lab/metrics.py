"""Transparent fixed-threshold matching, deliberately not the official KITTI AP protocol."""
import math

def area(b): return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def iou(a,b):
    overlap=intersection(a,b);union=area(a)+area(b)-overlap
    return overlap/union if union else 0.0

def match(detections, annotations, threshold=.5):
    if not math.isfinite(threshold) or not 0<threshold<=1: raise ValueError('Invalid IoU threshold')
    for d in detections:
        b=d['box']
        if len(b)!=4 or not all(math.isfinite(v) for v in b) or area(b)<=0 or not math.isfinite(d['score']):
            raise ValueError('Invalid detection')
    targets=[a for a in annotations if a['kind']=='Pedestrian']
    ignored=[a for a in annotations if a['kind'] in ('DontCare','Person_sitting','Cyclist')]
    used=set();records=[]
    for index in sorted(range(len(detections)),key=lambda i:(-detections[i]['score'],i)):
        detection=detections[index];b=detection['box']
        candidates=[(iou(b,t['box']),i) for i,t in enumerate(targets) if i not in used]
        best=max(candidates,default=(0,-1))
        if best[0]>=threshold:
            used.add(best[1]);status='TP';target=best[1]
        else:
            # Never hide duplicate detections of scored pedestrians in an ignore region.
            duplicate=any(iou(b,t['box'])>=threshold for t in targets)
            ignore=not duplicate and any((intersection(b,a['box'])/area(b)>.5 if a['kind']=='DontCare' else iou(b,a['box'])>=threshold) for a in ignored)
            status='IGNORED' if ignore else 'FP';target=None
        records.append({**detection,'index':index,'status':status,'target':target})
    missing=[{**t,'target':i} for i,t in enumerate(targets) if i not in used]
    return {'detections':records,'missed':missing,'tp':len(used),'fp':sum(d['status']=='FP' for d in records),
            'fn':len(missing),'ignored':sum(d['status']=='IGNORED' for d in records),
            'targets':len(targets),'matched_targets':sorted(used)}

def aggregate(frames):
    tp=sum(f['tp'] for f in frames);fp=sum(f['fp'] for f in frames);fn=sum(f['fn'] for f in frames)
    precision=tp/(tp+fp) if tp+fp else None;recall=tp/(tp+fn) if tp+fn else None
    return {'images':len(frames),'tp':tp,'fp':fp,'fn':fn,'ignored':sum(f['ignored'] for f in frames),
            'precision':precision,'recall':recall,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'false_positives_per_image':fp/len(frames) if frames else None}

def subgroup_recall(frames, annotations, predicate):
    total=found=0
    for f,anns in zip(frames,annotations):
        targets=[a for a in anns if a['kind']=='Pedestrian']
        chosen={i for i,a in enumerate(targets) if predicate(a)}
        total+=len(chosen);found+=len(chosen.intersection(f['matched_targets']))
    return {'found':found,'targets':total,'recall':found/total if total else None}

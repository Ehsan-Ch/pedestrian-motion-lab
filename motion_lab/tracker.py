"""Original constant-velocity Kalman tracker with gated global IoU assignment.

No labels, future frames, appearance embeddings or personal identities enter this module.
Time is measured in frames; reset Tracker between sequences.
"""
import math
import numpy as np
from .metrics import iou

def assign(cost):
    """Minimum cost one-to-one assignment for finite matrices with rows <= columns."""
    a=np.asarray(cost,dtype=float)
    if a.ndim!=2 or not np.isfinite(a).all() or a.shape[0]>a.shape[1]:
        raise ValueError('Expected finite rectangular cost matrix, rows <= columns')
    n,m=a.shape
    if not n:return []
    u=np.zeros(n+1);v=np.zeros(m+1);owner=np.zeros(m+1,dtype=int);previous=np.zeros(m+1,dtype=int)
    for row in range(1,n+1):
        owner[0]=row;column=0;best=np.full(m+1,np.inf);visited=np.zeros(m+1,dtype=bool)
        while True:
            visited[column]=True;active=owner[column];delta=np.inf;next_column=0
            for j in range(1,m+1):
                if visited[j]:continue
                reduced=a[active-1,j-1]-u[active]-v[j]
                if reduced<best[j]:best[j]=reduced;previous[j]=column
                if best[j]<delta:delta=best[j];next_column=j
            for j in range(m+1):
                if visited[j]:u[owner[j]]+=delta;v[j]-=delta
                else:best[j]-=delta
            column=next_column
            if owner[column]==0:break
        while column:
            parent=previous[column];owner[column]=owner[parent];column=parent
    return sorted((int(owner[j]-1),j-1) for j in range(1,m+1) if owner[j])

def measurement(box):
    x1,y1,x2,y2=box
    return np.array([(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1],dtype=float)

def box_from(state):
    cx,cy,w,h=state[:4];w=max(1.,w);h=max(1.,h)
    return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

class Track:
    def __init__(self,identity,detection,frame,mode):
        self.id=identity;self.mode=mode;self.x=np.r_[measurement(detection['box']),np.zeros(4)]
        self.p=np.diag([10.]*4+[100.]*4);self.hits=1;self.missed=0;self.age=1
        self.score=detection['score'];self.history=[self.x[:2].tolist()];self.first_frame=frame
    def predict(self):
        if self.mode=='motion':
            f=np.eye(8);f[:4,4:]=np.eye(4);self.x=f@self.x
            self.p=f@self.p@f.T+np.diag([1.,1.,1.,1.,.5,.5,.25,.25])
            self.x[2:4]=np.maximum(1.,self.x[2:4])
        self.missed+=1;self.age+=1
    def update(self,detection):
        z=measurement(detection['box'])
        if self.mode=='motion':
            h=np.eye(4,8);r=np.eye(4)*4
            k=np.linalg.solve(h@self.p@h.T+r,(self.p@h.T).T).T
            self.x+=k@(z-h@self.x)
            residual=np.eye(8)-k@h
            self.p=residual@self.p@residual.T+k@r@k.T
            self.p=(self.p+self.p.T)/2
            self.x[2:4]=np.maximum(1.,self.x[2:4])
        else:self.x[:4]=z
        self.hits+=1;self.missed=0;self.score=detection['score']
        self.history.append(self.x[:2].tolist());self.history=self.history[-20:]
    def snapshot(self):
        return {'id':self.id,'box':[round(float(v),3) for v in box_from(self.x)],'score':self.score,
                'state':'COASTING' if self.missed else 'OBSERVED','hits':self.hits,'missed':self.missed,
                'center':self.x[:2].round(3).tolist(),'velocity':self.x[4:6].round(3).tolist(),
                'history':[[round(v,3) for v in p] for p in self.history],
                'future':[(self.x[:2]+k*self.x[4:6]).round(3).tolist() for k in range(1,11)]}

class Tracker:
    def __init__(self,mode='motion',max_missed=5,min_hits=3,iou_gate=.2):
        if mode not in ('motion','last') or not isinstance(max_missed,int) or max_missed<0 or not isinstance(min_hits,int) or min_hits<1 or not math.isfinite(iou_gate) or not 0<iou_gate<=1:
            raise ValueError('Invalid tracker settings')
        self.mode=mode;self.max_missed=max_missed;self.min_hits=min_hits;self.gate=iou_gate
        self.tracks=[];self.next_id=1;self.last_frame=None
    def step(self,detections,frame):
        if not isinstance(frame,int) or frame<0 or (self.last_frame is not None and frame!=self.last_frame+1):
            raise ValueError('Frames must be nonnegative, consecutive and ordered')
        for d in detections:
            b=d['box']
            if len(b)!=4 or not all(math.isfinite(v) for v in b) or b[2]<=b[0] or b[3]<=b[1] or not math.isfinite(d['score']):
                raise ValueError('Invalid detection')
        self.last_frame=frame
        for track in self.tracks:track.predict()
        count=len(detections);cost=np.full((len(self.tracks),count+len(self.tracks)),1.01)
        for i,t in enumerate(self.tracks):
            for j,d in enumerate(detections):
                overlap=iou(box_from(t.x),d['box']);cost[i,j]=1-overlap if overlap>=self.gate else 1e6
        matched=set()
        for i,j in assign(cost):
            if j<count and cost[i,j]<1.01:self.tracks[i].update(detections[j]);matched.add(j)
        self.tracks=[t for t in self.tracks if t.missed<=self.max_missed]
        for j,d in enumerate(detections):
            if j not in matched:
                self.tracks.append(Track(self.next_id,d,frame,self.mode));self.next_id+=1
        return [t.snapshot() for t in self.tracks if t.hits>=self.min_hits]

import itertools
import json
import unittest
from pathlib import Path
import numpy as np
from motion_lab.tracker import Tracker,assign
from motion_lab.evaluate import identity_switches,forecast_errors,summarize_forecasts
from motion_lab.data import parse_labels

def detection(x=0,y=0):return {'box':[x,y,x+20,y+50],'score':.9}

class AssignmentTests(unittest.TestCase):
    def test_global_solution_beats_greedy(self):
        self.assertEqual(assign([[.1,.2],[.11,1.]]),[(0,1),(1,0)])
    def test_random_rectangles_equal_brute_force(self):
        rng=np.random.default_rng(42)
        for n in range(1,5):
            for m in range(n,6):
                a=rng.uniform(-1,2,(n,m));pairs=assign(a)
                best=min(sum(a[i,j] for i,j in enumerate(p)) for p in itertools.permutations(range(m),n))
                self.assertAlmostEqual(sum(a[i,j] for i,j in pairs),best)
    def test_empty_and_invalid(self):
        self.assertEqual(assign(np.empty((0,0))),[])
        for a in ([[float('nan')]],[[1],[2]],[1,2]):
            with self.assertRaises(ValueError):assign(a)

class TrackerTests(unittest.TestCase):
    def test_confirmation_requires_three_observations(self):
        t=Tracker();self.assertEqual(t.step([detection()],0),[]);self.assertEqual(t.step([detection()],1),[])
        self.assertEqual(t.step([detection()],2)[0]['hits'],3)
    def test_constant_velocity_forecast(self):
        t=Tracker()
        for f in range(20):out=t.step([detection(f*3)],f)
        self.assertEqual(out[0]['id'],1);self.assertAlmostEqual(out[0]['velocity'][0],3,delta=.1)
        self.assertAlmostEqual(out[0]['future'][4][0],10+24*3,delta=.5)
    def test_coasting_and_reacquisition(self):
        t=Tracker()
        for f in range(5):t.step([detection(f*2)],f)
        for f in range(5,8):self.assertEqual(t.step([],f)[0]['state'],'COASTING')
        r=t.step([detection(16)],8);self.assertEqual(r[0]['id'],1);self.assertEqual(r[0]['state'],'OBSERVED')
    def test_expiration_does_not_reuse_id(self):
        t=Tracker(min_hits=1,max_missed=2);self.assertEqual(t.step([detection()],0)[0]['id'],1)
        self.assertTrue(t.step([],1));self.assertTrue(t.step([],2));self.assertEqual(t.step([],3),[])
        self.assertEqual(t.step([detection()],4)[0]['id'],2)
    def test_nonoverlap_creates_new_track(self):
        t=Tracker(min_hits=1,max_missed=0);t.step([detection()],0)
        self.assertEqual(t.step([detection(1000)],1)[0]['id'],2)
    def test_detection_order_does_not_switch_ids(self):
        t=Tracker(min_hits=1);t.step([detection(0),detection(100)],0)
        out=t.step([detection(102),detection(2)],1)
        self.assertLess(out[0]['center'][0],out[1]['center'][0]);self.assertEqual([p['id'] for p in out],[1,2])
    def test_last_position_baseline_has_zero_velocity(self):
        t=Tracker('last',min_hits=1)
        for f in range(8):out=t.step([detection(f*2)],f)
        self.assertEqual(out[0]['future'],[out[0]['center']]*10)
    def test_sequence_reset(self):
        a=Tracker(min_hits=1);b=Tracker(min_hits=1)
        self.assertEqual(a.step([detection()],0),b.step([detection()],0))
    def test_rejects_reversed_missing_and_duplicate_frames(self):
        for next_frame in (0,2,-1):
            t=Tracker();t.step([],0)
            with self.assertRaises(ValueError):t.step([],next_frame)
    def test_invalid_input_does_not_advance_state(self):
        t=Tracker();bad=detection();bad['score']=float('nan')
        with self.assertRaises(ValueError):t.step([bad],0)
        self.assertIsNone(t.last_frame);t.step([],0)
    def test_snapshots_are_not_mutated_by_later_frames(self):
        t=Tracker(min_hits=1);a=t.step([detection()],0);before=json.dumps(a)
        for f in range(1,12):t.step([detection(f)],f)
        self.assertEqual(json.dumps(a),before)
    def test_covariance_stays_positive(self):
        t=Tracker()
        for f in range(100):t.step([detection(f*.3,f*.1)],f)
        self.assertGreater(np.linalg.eigvalsh(t.tracks[0].p).min(),0)
    def test_invalid_parameters(self):
        for args in ({'min_hits':0},{'max_missed':-1},{'iou_gate':0},{'mode':'unknown'}):
            with self.assertRaises(ValueError):Tracker(**args)

class EvaluationTests(unittest.TestCase):
    def test_identity_switch_counts_return_after_gap(self):
        def frame(f,identity):return {'frame':f,'targets_detail':[{'identity':7}],'detections':[{'status':'TP','target':0,'id':identity}]}
        events=identity_switches([frame(0,1),{'frame':1,'detections':[]},frame(2,2),frame(3,2)])
        self.assertEqual(events,[{'frame':2,'target':7,'before':1,'after':2}])
    def test_forecasts_compare_same_anchor_to_future_truth(self):
        frames=[]
        for f in range(6):
            ann={'kind':'Pedestrian','identity':1,'box':[f,0,f+20,50]}
            tracks=[{'status':'TP','state':'OBSERVED','target':0,'id':1,'center':[10,25],'future':[[10+k,25] for k in range(1,11)]}] if f==0 else []
            frames.append({'frame':f,'annotations':[ann],'variants':{'motion':{'detections':tracks}}})
        r=forecast_errors(frames,5);self.assertEqual(len(r),1);self.assertEqual(r[0]['motion_fde'],0);self.assertEqual(r[0]['stationary_fde'],5)
        frames[-1]['annotations']=[];self.assertEqual(forecast_errors(frames,5),[])
    def test_no_eligible_forecasts_are_null(self):
        self.assertIsNone(summarize_forecasts([])['motion_fde'])
    def test_tracking_label_fields(self):
        row='0 4 Pedestrian 0 1 0 10 20 30 80 1 1 1 0 0 0 0'
        a=parse_labels(row)[0][0];self.assertEqual(a['identity'],4);self.assertEqual(a['box'],[10,20,30,80])
        with self.assertRaises(ValueError):parse_labels('bad row')

class RecordedResultsTests(unittest.TestCase):
    def test_recorded_metrics_and_forecasts_reconstruct(self):
        from motion_lab.metrics import aggregate
        r=json.loads(Path('reports/results.json').read_text())
        self.assertEqual([c['id'] for c in r['clips']],['0011','0013'])
        for c in r['clips']:self.assertEqual([f['frame'] for f in c['frames']],list(range(60)))
        for method in ('motion','last'):
            metric=aggregate([f['variants'][method] for c in r['clips'] for f in c['frames']])
            for key,value in metric.items():self.assertEqual(value,r['summary'][method][key])
        for h in (5,10):
            records=[v for c in r['clips'] for v in forecast_errors(c['frames'],h)]
            for key,value in summarize_forecasts(records).items():
                self.assertAlmostEqual(value,r['forecast_summary'][str(h)][key],places=8)

if __name__=='__main__':unittest.main()

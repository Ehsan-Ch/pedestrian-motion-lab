import json
import math
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from motion_lab.detector import letterbox,decode,Detector
from motion_lab.metrics import iou,match,aggregate,subgroup_recall

def annotation(box=(0,0,10,20),kind='Pedestrian',occlusion=0):
    return {'box':list(box),'kind':kind,'occlusion':occlusion,'truncation':0}
def detection(box=(0,0,10,20),score=.9): return {'box':list(box),'score':score}

class MatchingTests(unittest.TestCase):
    def test_iou_geometry_and_disjoint(self):
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1)
        self.assertAlmostEqual(iou([0,0,10,10],[5,0,15,10]),1/3)
        self.assertEqual(iou([0,0,1,1],[2,2,3,3]),0)
    def test_perfect_and_duplicate_detection(self):
        r=match([detection(),detection(score=.8)],[annotation()])
        self.assertEqual((r['tp'],r['fp'],r['fn']),(1,1,0))
    def test_high_confidence_detection_gets_first_match(self):
        r=match([detection(score=.6),detection(score=.9)],[annotation()])
        self.assertEqual(r['detections'][0]['index'],1)
        self.assertEqual(r['detections'][1]['status'],'FP')
    def test_missed_target_and_unmatched_detection(self):
        r=match([detection([30,30,40,50])],[annotation()])
        self.assertEqual((r['tp'],r['fp'],r['fn']),(0,1,1))
    def test_dontcare_uses_detection_area(self):
        r=match([detection()],[annotation([0,0,100,100],'DontCare')])
        self.assertEqual((r['ignored'],r['fp']),(1,0))
    def test_person_match_precedes_ignore(self):
        r=match([detection(),detection(score=.8)],[annotation(),annotation([0,0,100,100],'DontCare')])
        self.assertEqual((r['tp'],r['fp'],r['ignored']),(1,1,0))
    def test_cyclist_and_sitting_person_are_not_scored_targets(self):
        for kind in ('Cyclist','Person_sitting'):
            r=match([detection()],[annotation(kind=kind)])
            self.assertEqual((r['targets'],r['ignored']),(0,1))
    def test_no_predictions_is_not_perfect_precision(self):
        s=aggregate([match([],[annotation()])]);self.assertIsNone(s['precision']);self.assertEqual(s['recall'],0)
        empty=aggregate([match([],[])]);self.assertIsNone(empty['precision']);self.assertIsNone(empty['recall'])
    def test_subgroups_use_annotation_identity(self):
        anns=[annotation(occlusion=1),annotation([50,0,60,50])]
        r=match([detection()],anns);s=subgroup_recall([r],[anns],lambda a:a['occlusion']>0)
        self.assertEqual(s,{'found':1,'targets':1,'recall':1})
    def test_nonfinite_or_degenerate_detection_rejected(self):
        for d in (detection([0,0,0,1]),detection(score=math.nan),detection([0,0,1,math.inf])):
            with self.assertRaises(ValueError):match([d],[])

class DetectorTests(unittest.TestCase):
    def test_letterbox_color_order_scale_and_padding(self):
        image=np.zeros((100,200,3),np.uint8);image[:]=[5,10,20]
        blob,scale=letterbox(image);self.assertEqual(blob.shape,(1,3,640,640));self.assertEqual(scale,3.2)
        np.testing.assert_array_equal(blob[0,:,0,0],[20,10,5]);np.testing.assert_array_equal(blob[0,:,500,0],[114]*3)
    def test_invalid_image_contract(self):
        for image in (None,np.zeros((100,100)),np.zeros((100,100,3)),np.zeros((10,10,3),np.uint8)):
            with self.assertRaises(ValueError):letterbox(image)
    def raw(self):
        a=np.zeros((1,8400,85),np.float32)
        a[0,0,:4]=[20,20,math.log(10),math.log(20)]
        a[0,0,4:6]=[.9,.9]
        return a
    def test_decoding_recovers_box_and_objectness_product(self):
        det=decode(self.raw(),1,640,640)
        self.assertEqual(len(det),1);np.testing.assert_allclose(det[0]['box'],[120,80,200,240],atol=.01)
        self.assertAlmostEqual(det[0]['score'],.81,places=5)
    def test_other_best_class_is_not_reported_as_person(self):
        a=self.raw();a[0,0,6]=.95;self.assertEqual(decode(a,1,640,640),[])
    def test_nms_removes_duplicate_boxes(self):
        a=self.raw();a[0,1]=a[0,0];a[0,1,0]-=1
        self.assertEqual(len(decode(a,1,640,640)),1)
    def test_padding_only_detection_is_removed(self):
        a=self.raw();a[0,0,1]=70
        self.assertEqual(decode(a,1,640,100),[])
    def test_no_scores_and_nonfinite_outputs(self):
        self.assertEqual(decode(np.zeros((1,8400,85)),1,640,640),[])
        a=self.raw();a[0,5,3]=math.nan
        with self.assertRaises(ValueError):decode(a,1,640,640)
    def test_incorrect_model_hash_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fake.onnx';p.write_bytes(b'not a model')
            with self.assertRaisesRegex(ValueError,'checksum'):Detector(p)

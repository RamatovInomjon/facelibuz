# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division

import glob
import os.path as osp

import numpy as np
import onnxruntime
from numpy.linalg import norm

from ..model_zoo import model_zoo
from .common import Face
from facelibuz.utils.sort_tracker import SORT
from time import time
from facelibuz.utils.trackableobject import TrackableObject
__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, name='s', root='~/./facelibuz', allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = osp.join(root, name)        #ensure_available('models', name, root=root)
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']
        self.tracker = None
        if 'tracking' in allowed_modules:
            self.tracker = SORT(max_lost=5, iou_threshold=0.3)
            self.trackableObjects = {}



    def prepare(self, ctx_id, det_thresh=0.7, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        start = time()
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        rects = []
        kps_list = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            if self.tracker is None:
                
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                for taskname, model in self.models.items():
                    if taskname=='detection':
                        continue
                    # start = time()
                    model.get(img, face)
                ret.append(face)
            else:
                landmarks = np.array(kps)
                landmarks = np.transpose(landmarks).reshape(10, -1)
                landmarks = np.transpose(landmarks)[0]
                x1,y1,x2,y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                kps = landmarks.astype('int')
                rects.append([x1, y1, x2, y2])
                kps_list.append(kps)
            
        if self.tracker:
            objects = self.tracker.update(np.array(rects), np.array(kps_list), np.ones(len(rects)))
            for i in range(len(self.trackableObjects)):
                track_obj = self.trackableObjects[i]
                track_obj.live = False
                track_obj.lost_count+=1
            for obj in objects:
                objectID = obj[1]
                bbox = obj[2:6]
                kps = obj[6]                
                facial5points = [[kps[j], kps[j + 5]] for j in range(5)]
                
                to = self.trackableObjects.get(objectID, None)
                
                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, obj)
                face = Face(bbox=np.array(bbox), kps=np.array(facial5points), det_score=1) 
                to.bbox = np.array(bbox)
                to.kps = np.array(facial5points)
                to.live = True
                to.lost_count=0
                if not to.recognized:
                    self.models['recognition'].get(img, face)                  
                    to.embeddings = face.embedding
                    

                self.trackableObjects[objectID] = to
            # for i in range(len(self.trackableObjects)):
            #     track_obj = self.trackableObjects[i]
            #     if not track_obj.live:
            #         track_obj.lost_count+=1
            #     if track_obj.lost_count>lost_count:
            #         del self.trackableObjects[i]

            

        return ret

    def draw_on(self, img, faces, lost_count=30):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            if not face.live and face.lost_count>lost_count:
                continue
            if face.bbox is not None:
                box = face.bbox.astype(np.int32)
            else:
                continue
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.recognized:
                text = "{}".format(face.name)
            else:
                text = "ID  {}".format(face.objectID)
            cv2.putText(dimg, text, (box[0], box[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
            # if face.kps is not None:
            #     kps = face.kps.astype(np.int32)
            #     #print(landmark.shape)
            #     for l in range(kps.shape[0]):
            #         color = (0, 0, 255)
            #         if l == 0 or l == 3:
            #             color = (0, 255, 0)
            #         cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
            #                    2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

            #for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(np.int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg


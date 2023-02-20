import os
import sys
from datetime import datetime
from time import sleep, time
import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
from hikvisionapi import Client
from PIL import Image
from scipy.spatial.distance import cosine
from tqdm import tqdm

from align.align_trans import get_reference_facial_points, warp_and_crop_face
from insightface.app import FaceAnalysis
from trackableobject import TrackableObject

sys.path.append('utils/')
from sort_tracker import SORT

reference = get_reference_facial_points(default_square=True)

known_people_list = []
model = ort.InferenceSession("models/ir_50_arcface_batch384.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

global cam
source = 1
def load_known_faces():
    global known_people_list, model
    # if os.path.exists("facedb.npy"):
    #     known_people_list = list(np.load('facedb.npy', allow_pickle=True))
    #     print("loaded from file")
        # print(known_people_list)
        # sleep(3)
    known_faces="test/"
    list_images = os.listdir(known_faces)
    for img_name in tqdm(list_images):
        person_name = img_name.split(".")[0]
        image_path = os.path.join(known_faces, img_name)
        image = cv2.imread(image_path)
        face_encodings = app.get(image)
        if len(face_encodings)==0:
            print("Cannot extract face feature from : {}".format(img_name))
            continue;
        person = {}
        person["name"] = person_name
        bbox = face_encodings[0].bbox
        # print(face.kps)
        landmarks = np.array(face_encodings[0].kps)
        landmarks = np.transpose(landmarks).reshape(10, -1)
        landmarks = np.transpose(landmarks)[0]
        facial5points = [[landmarks[j], landmarks[j + 5]] for j in range(5)]
        face_img =  align_face_onnx(image, facial5points)
        embeddings = get_embeddings_immediate_onnx(face_img, model)
        person["embedding"] = embeddings
        known_people_list.append(person)
    np.save('facedb.npy', known_people_list)    


def key_func(person):
    return person["sim"]


def resize_frame(frame):
    frame = cv2.resize(frame, (640*1024//480, 1024))

    frame = frame[:, frame.shape[1]//2 - 300: frame.shape[1] // 2 + 300]

    return frame


def find_face(embedding):
    best_matches = []
    
    for person in known_people_list:
        sim = 1 - cosine(person["embedding"], embedding)
        if(sim>0.45):
            known = {}
            known["person"]=person
            known["sim"]=sim
            best_matches.append(known)
    if len(best_matches) == 0:
        return 
    best_matches.sort(key=key_func)
    # print(best_matches[0]["sim"])
    return best_matches[0]

def get_frame_hik():
    
    vid = cam.Streaming.channels[101].picture(method ='get', type = 'opaque_data')
    bytes = b''
    # with open('screen.jpg', 'wb') as f:
    for chunk in vid.iter_content(chunk_size=1024):

        bytes += chunk
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame


def get_frame_video():
    
    ret, frame = cam.read()
    # print(ret)
    return frame
def align_face_onnx(img, landmark5):
    
    warped_face = warp_and_crop_face(img, landmark5, reference, crop_size=(112, 112), )

    img_warped = Image.fromarray(warped_face)
    # cv2.imwrite("test.jpg", warped_face)
    return img_warped


def get_embeddings_immediate_onnx(face_img, model, input_size=None):
    if input_size is None:
        input_size = [112, 112]

    if face_img is None:
        return

    transform = transforms.Compose(
        [
            transforms.Resize(
                [int(128 * input_size[0] / 112),
                 int(128 * input_size[0] / 112)],
            ),  # smaller side resized
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
    )

    # apply transformations  
    face_img = transform(face_img)
    face_img = face_img[None, ...]
    face_img = face_img.cpu().detach().numpy()

    outputs = None

    outputs = model.run(None, {'input.1': face_img})



    if outputs is None:
        return

    return outputs[0]

import io
import json

file_name = "{}_exp.mp4".format(datetime.now())
my_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), 25, (1280, 720))
def main():

    


    load_known_faces()

    # ct = CentroidTracker(maxDisappeared=30, maxDistance=200)
    trackers = []
    ct = SORT(max_lost=5, iou_threshold=0.3)
    trackableObjects = {}
    global model
    # np.save('test3.npy', known_people_listp00)    
    # exit(0)
    # d = np.load('test3.npy')
    # print(known_people_list)
    if False:
        global cam
        cam = Client('http://192.168.0.250', 'admin', 'inomjon199303_R')
    # else:
    # rtsp://admin:inomjon199303_R@192.168.0.250:554/Streaming/channels/101
    cam = cv2.VideoCapture("videos/D15_20230213153059.mp4")
    Y_line = 720
    totalDown = 0
    totalUp = 0
    count = 0
    while(1):
        # _, frame = cap.read()

        # cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # print(_)
        rects=[]
        if False:
            frame = get_frame_hik()
            # print("hi")
        # else:
        frame = get_frame_video()
        count+=1
        if count<1700:
            continue
        # frame = cv2.resize(frame, )
        if frame is None:
            print("frame is None")
            # continue
            break

        start = time()
        # print(frame.shape)
        faces = app.get(frame)
        kps_list = []
        # print("time cost: {}".format(time()-start))
        for face in faces:
            bbox = face.bbox
            # print(face.kps)
            landmarks = np.array(face.kps)
            landmarks = np.transpose(landmarks).reshape(10, -1)
            landmarks = np.transpose(landmarks)[0]
            # print(landmarks.astype('int'))
            # exit(0)
            person = None
            x1,y1,x2,y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            kps = landmarks.astype('int')
            rects.append([x1, y1, x2, y2])
            kps_list.append(kps)
            # print(face.kps)
            # exit(0)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
            # print(face)
            # person = find_face(face.embedding)

            # if person is None:
            #     continue
            # frame = cv2.putText(frame, person["person"]["name"], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        #     # print("recognized as: {}, similarity: {}".format(person["person"]["name"], person["sim"]))
        #     # img_name = "{}_{}.jpg".format(person["person"]["name"], person["sim"])
        #     # img_path = os.path.join("results", img_name)
        #     # cv2.imwrite(img_path, frame)
        # print(rects)
        
        # objects = ct.update(rects)
        # print(rects)
        # cv2.line(frame, (0, Y_line), (2560, Y_line), (0, 0, 0), 3)
        objects = ct.update(np.array(rects), np.array(kps_list), np.ones(len(rects)))
        # print(objects)
        # loop over the tracked objects
        tic = time()
        for obj in objects:
            # check to see if a trackable object exists for the current
            # object ID
            # print(obj)
            objectID = obj[1]
            cv2.rectangle(frame, (obj[2], obj[3]), (obj[4], obj[5]), (0,255,0), 2)
            bbox = obj[2:6]
            kps = obj[6]
            cy = (obj[3]+obj[5])/2
            # print(centroid[6:])
            # print(kps)
            # print(len(kps))
            facial5points = [[kps[j], kps[j + 5]] for j in range(5)]
            # print(facial5points)
            # cv2.circle(frame, (facial5points[0][0],facial5points[0][1]), 4, (0,0,255), -1)
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, obj)
            else:
                to.path_length+=1
                to.centroids.append(obj)
                if not to.counted:
                    if to.startY > Y_line and cy < Y_line and to.path_length>2:
                        totalUp += 1
                        to.counted = True

                    elif to.startY < Y_line and cy > Y_line and to.path_length>2:
                        totalDown += 1
                        to.counted = True

                if not to.recognized:
                    person = None
                    face_img =  align_face_onnx(frame, facial5points)
                    embeddings = get_embeddings_immediate_onnx(face_img, model)
                    person = find_face(embeddings)
                    if person is not None:
                        print(person["person"]["name"])
                        to.name = person["person"]["name"]
                        to.recognized = True

                
                        

            trackableObjects[objectID] = to
            if to.recognized:
                text = "{}".format(to.name)
            else:
                text = "ID  {}".format(objectID)
            cv2.putText(frame, text, (obj[2], obj[3]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
            # cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        # print(time()-start)
        # print("loop: ", time()-tic)
        info = [
        ("Exit", totalUp),
        ("Enter", totalDown),
        ]


        # # Display the output
        # for (i, (k, v)) in enumerate(info):
        #     text = "{}: {}".format(k, v)
        #     cv2.putText(frame, text, (10, 1440 - ((i * 40) + 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)

        cv2.imshow("SORT tracker", cv2.resize(frame, (1080,720)))
        my_writer.write(cv2.resize(frame, (1280, 720)))
        k = cv2.waitKey(2)
        if k==ord("q"):
            cv2.destroyAllWindows()
            # cap.release()
            break
    




def test():
    img1 = cv2.imread("1.jpeg")
    start = time()
    faces = app.get(img1)
    # print("time cost: {}".format(time()-start))
    for face in faces:
        print(face.rec_score)
        bbox = face.bbox
        person = None
        x1,y1,x2,y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
    
    cv2.imshow("test", cv2.resize(img1, (480,640)))

    cv2.waitKey(0)








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default=0, help="input video/camera/rtsp stream")
    args = parser.parse_args()
    # ap.add_argument("-d_w", "--detection_weights", default="weights/helmet_head_person_s.pt", help="detection model weights")
    # ap.add_argument("-d", "--device", default="cpu", help="device type, cuda:0 or cpu")
    # ap.add_argument("-th", "--conf_thresh", default=0.25, help="confidence threshold")
    # ap.add_argument("-is_video", "--is_video", default=False, help="Read from hikvision api")
    app = FaceAnalysis(root='./models', name='my_combo', allowed_modules=['detection'])
    app.prepare(ctx_id=0)
    # test()
    main()
    # img = ins_get_image('test')
    # faces = app.get(img)
    # for face in faces:
    #     print(face)

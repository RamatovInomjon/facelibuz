import argparse
import cv2
from facelibuz.app import FaceAnalysis
import depthai as dai








def main():
    source = args.source
    cap = cv2.VideoCapture(source)
    from time import time
    if args.oak:
        pipeline = dai.Pipeline()
        # Define source and output
        camRgb = pipeline.createColorCamera()
        xoutVideo = pipeline.createXLinkOut()

        xoutVideo.setStreamName("video")

        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        camRgb.setVideoSize(3840,2160)
        camRgb.setFps(30)
        xoutVideo.input.setBlocking(False)
        xoutVideo.input.setQueueSize(10)

        # Linking
        camRgb.video.link(xoutVideo.input)

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            video = device.getOutputQueue(name="video", maxSize=10, blocking=False)


            while True:
                videoIn = video.get()
                frame = videoIn.getCvFrame()
                
                start = time()
                faces = app.get(frame)
                if len(faces)==0:
                    faces = app.trackableObjects
                    # print(type(faces))
                    # print(len(faces))
                    # print(faces[0].objectID, faces[0].name, faces[0].embeddings)
                    # for face in faces:
                    #     print(face)
                # print("get: ", time()-start)
                
                frame = app.draw_on(frame, faces)
                # cv2.resize(frame, (1280, 720))
                cv2.imshow("SORT tracker", cv2.resize(frame, (1280, 720)))
                # my_writer.write(cv2.resize(frame, (1280, 720)))
                k = cv2.waitKey(2)
                if k==ord("q"):
                    cv2.destroyAllWindows()
                    cap.release()
                    break
    else:
        while True:
            
            ret, frame = cap.read()
            if not ret:
                print("frame is None")
                break
            start = time()
            faces = app.get(frame)
            if len(faces)==0:
                faces = app.trackableObjects
                
            frame = app.draw_on(frame, faces)
            # cv2.resize(frame, (1280, 720))
            cv2.imshow("SORT tracker", cv2.resize(frame, (1280, 720)))
            # my_writer.write(cv2.resize(frame, (1280, 720)))
            k = cv2.waitKey(2)
            if k==ord("q"):
                cv2.destroyAllWindows()
                cap.release()
                break


        




    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default=0, help="input video/camera/rtsp stream")
    parser.add_argument('-mt', "--model-type", default='s', help="model type, options: s,m,l")
    parser.add_argument('-mp', '--model-path', default='./models', help="models path, s,m or l folders")
    parser.add_argument("-oak", "--oak", default=False, type=bool)
    args = parser.parse_args()

    # if args.oak:
    #     # print("hi")
    #     # import pyvirtualcam
    #     import depthai as dai
    #     # Create pipeline
    #     pipeline = dai.Pipeline()
    #     cam = pipeline.create(dai.node.ColorCamera)
    #     cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    #     # cam.setPreviewSize(1280,720)
    #     xout = pipeline.create(dai.node.XLinkOut)
    #     xout.setStreamName("rgb")
    #     cam.preview.link(xout.input)
    #     with dai.Device(pipeline) as device:
    #         qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    app = FaceAnalysis(root='./models', name='l', allowed_modules=['detection', 'recognition', 'tracking'])
    app.prepare(ctx_id=0)
    
    
    main()



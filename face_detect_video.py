import cv2
import numpy as np
import os



PROTOTXT ="/home/raj/development/deepnet/deploy.prototxt.txt" 
MODEL = "/home/raj/development/deepnet/res10_300x300_ssd_iter_140000.caffemodel"
VIDEO_PATH = "/home/raj/development/deepnet/images/vid_clip.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
out = cv2.VideoWriter('60fps.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 100, (854,460))
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
while (True):
    ret, image = cap.read()
    #print(type(image))
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (xmin, ymin, xmax, ymax) = box.astype("int")
            y = ymin - 10 if ymin - 10 > 10 else ymin + 10
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax),(0, 255, ), 2)
    out.write(image)
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


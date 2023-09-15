import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort import *
from helper import create_video_writer

cap = cv2.VideoCapture("videos/sample.mp4")  # For Video
writer = create_video_writer(cap, "Output.mp4")


model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("Images/mask.png")

# Tracking
tracker = Sort(max_age=20)

limitsDown = [150, 220, 250, 220]
limitsUp= [70, 170, 160, 170]
totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            print(box)
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 6)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 , id= int(x1), int(y1), int(x2), int(y2) ,int(id)
        print(result)
        w, h = x2 - x1, y2 - y1

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Using cv2.putText() method
        image = cv2.putText(img, currentClass, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 0), 1, cv2.LINE_AA)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)


    cv2.putText(img,f'Up: {len(totalCountUp)}',(480,40),cv2.FONT_HERSHEY_PLAIN,2,(139,195,75),3)
    cv2.putText(img, f'Down: {len(totalCountDown)}', (480,70), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 230), 3)
    cv2.imshow("Video", img)
    writer.write(img)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
writer.release()
cv2.destroyAllWindows()
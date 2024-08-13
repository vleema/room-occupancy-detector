import cv2
import pandas as pd
import numpy as np
import argparse
from ultralytics import YOLO

# construct the argument parser and parse the arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-v", "--video", type=str, help="video source (none: webcam)")
args = vars(arg_parser.parse_args())

if not args.get("video", False):
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(args["video"])

yolo_model = YOLO("yolov8s.pt")

area1_coordinates = [(312, 388), (289, 390), (474, 469), (497, 462)]

area2_coordinates = [(279, 392), (250, 397), (423, 477), (454, 469)]


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

coco_file = open("coco.txt", "r")
coco_data = coco_file.read()
class_list = coco_data.split("\n")
# print(class_list)

frame_count = 0


while True:
    read_successful, current_frame = video_capture.read()
    if not read_successful:
        break
    frame_count += 1
    if frame_count % 2 != 0:
        continue
    current_frame = cv2.resize(current_frame, (1020, 500))
    #    current_frame=cv2.flip(current_frame,1)
    yolo_results = yolo_model.predict(current_frame)
    #   print(results)
    bounding_boxes = yolo_results[0].boxes.data
    bounding_boxes_df = pd.DataFrame(bounding_boxes).astype("float")
    #    print(px)
    list = []

    for index, row in bounding_boxes_df.iterrows():
        #        print(row)
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        class_index = int(row[5])
        class_name = class_list[class_index]
        if "person" in class_name:
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                current_frame,
                str(class_name),
                (x1, y1),
                cv2.FONT_HERSHEY_COMPLEX,
                (0.5),
                (255, 255, 255),
                1,
            )

    # # Draw area 1
    # cv2.polylines(
    #     current_frame, [np.array(area1_coordinates, np.int32)], True, (255, 0, 0), 2
    # )
    # cv2.putText(
    #     current_frame,
    #     str("1"),
    #     (504, 471),
    #     cv2.FONT_HERSHEY_COMPLEX,
    #     (0.5),
    #     (0, 0, 0),
    #     1,
    # )
    #
    # # Draw area 2
    # cv2.polylines(
    #     current_frame, [np.array(area2_coordinates, np.int32)], True, (255, 0, 0), 2
    # )
    # cv2.putText(
    #     current_frame,
    #     str("2"),
    #     (466, 485),
    #     cv2.FONT_HERSHEY_COMPLEX,
    #     (0.5),
    #     (0, 0, 0),
    #     1,
    # )

    cv2.imshow("RGB", current_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()

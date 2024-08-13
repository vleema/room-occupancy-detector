import cv2
import pandas as pd
import numpy as np
import argparse
import sys
from ultralytics import YOLO
from tracker import Tracker

# construct the argument parser and parse the arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-v", "--video", type=str, help="video source (none: webcam)")
args = vars(arg_parser.parse_args())

if not args.get("video", False):
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(args["video"])

yolo_model = YOLO("yolov8s.pt")

inner_area_coordinates = [(612, 426), (594, 418), (720, 380), (753, 384)]

outer_area_coordinates = [(629, 437), (645, 445), (774, 397), (760, 390)]


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR, file=sys.stderr)


cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

coco_file = open("coco.txt", "r")
coco_data = coco_file.read()
class_list = coco_data.split("\n")

frame_count = 0
person_tracker = Tracker()
persons_entering = {}
entering = set()

persons_exiting = {}
leaving = set()
while True:
    read, current_frame = video_capture.read()
    if not read:
        break
    frame_count += 1
    if frame_count % 2 != 0:
        continue
    current_frame = cv2.resize(current_frame, (1020, 500))
    yolo_results = yolo_model.predict(current_frame)
    bounding_boxes = yolo_results[0].boxes.data
    bounding_boxes_df = pd.DataFrame(bounding_boxes).astype("float")

    coordinates_list = []

    for index, row in bounding_boxes_df.iterrows():
        class_index = int(row[5])
        class_name = class_list[class_index]

        if "person" in class_name:
            coordinates_list.append(
                [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
            )

    boxes_ids = person_tracker.update(coordinates_list)
    for box_id in boxes_ids:
        x1, y1, x2, y2, person_id = box_id
        person_downward_left = (x2, y2)
        person_in_inner_area = cv2.pointPolygonTest(
            np.array(inner_area_coordinates), (x2, y2), False
        )
        # Actually is a number so we need to check if it is greater than 0
        if person_in_inner_area >= 0:
            persons_entering[person_id] = person_downward_left
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if person_id in persons_entering:
            person_in_outer_area = cv2.pointPolygonTest(
                np.array(outer_area_coordinates), (x2, y2), False
            )
            if person_in_outer_area >= 0:
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                entering.add(person_id)

        person_in_outer_area = cv2.pointPolygonTest(
            np.array(outer_area_coordinates), (x2, y2), False
        )
        if person_in_outer_area >= 0:
            persons_exiting[person_id] = person_downward_left
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if person_id in persons_exiting:
            person_in_inner_area = cv2.pointPolygonTest(
                np.array(inner_area_coordinates), (x2, y2), False
            )
            if person_in_inner_area >= 0:
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                leaving.add(person_id)

    # Display the amount of people entering and leaving
    # cv2.putText(
    #     current_frame,
    #     "Entered: " + str(len(entering)),
    #     (60, 80),
    #     cv2.FONT_HERSHEY_COMPLEX,
    #     (0.5),
    #     (0, 0, 255),
    #     1,
    # )
    # cv2.putText(
    #     current_frame,
    #     "Leaved: " + str(len(leaving)),
    #     (60, 100),
    #     cv2.FONT_HERSHEY_COMPLEX,
    #     (0.5),
    #     (0, 255, 0),
    #     1,
    # )
    #

    # Draw the inner area
    cv2.polylines(
        current_frame, [np.array(inner_area_coordinates)], True, (0, 0, 255), 2
    )
    # Draw the outer area
    cv2.polylines(
        current_frame, [np.array(outer_area_coordinates)], True, (0, 255, 0), 2
    )

    cv2.putText(
        current_frame,
        "Total de pessoas na sala: " + str(len(entering) - len(leaving)),
        (60, 60),
        cv2.FONT_HERSHEY_COMPLEX,
        (0.5),
        (255, 255, 0),
        1,
    )

    cv2.imshow("RGB", current_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()

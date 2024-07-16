

# importing the necessary libraries
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision # this 'vision' module is the most important to process our object detection
import numpy as np


# callback function which will be used in the options ahead in the code
def visualize_callback(result: vision.ObjectDetectorResult,
                         output_image: mp.Image, timestamp_ms: int):
      result.timestamp_ms = timestamp_ms
      detection_result_list.append(result)


# this function takes an image and just process it
# by process I mean that it will put the name of object detected by our model, the probablity and a bounding box around it

# arguments: 
# img - the actual image
# detection_result - this is a mediapipe object which contains information about our detected objects
def visualize(img, detection_result):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        cv.rectangle(img, (bbox.origin_x, bbox.origin_y), (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (0,255,0), 2)


        category = detection.categories[0]
        category_name = category.category_name
        probaScore = round(category.score, 2)
        txt = category_name + '(' + str(probaScore) + ')'
        cv.putText(img, txt, (bbox.origin_x, bbox.origin_y + 20), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
    return img





baseoptions = python.BaseOptions(model_asset_path='D:\Zishan\Projects\Object Detection using Mediapipe\efficientdet_lite0.tflite')
VisionRunningMode = mp.tasks.vision.RunningMode
options = vision.ObjectDetectorOptions(base_options=baseoptions, running_mode = VisionRunningMode.LIVE_STREAM, score_threshold=0.5, max_results=5, result_callback=visualize_callback)
detector = vision.ObjectDetector.create_from_options(options)


detection_result_list = []


cap = cv.VideoCapture(0)
counter = 0
while True:
    res, frame = cap.read()
    frame = cv.flip(frame, 1)
    counter += 1
    rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detector.detect_async(mp_image, counter)
    current_frame = mp_image.numpy_view()
    current_frame = cv.cvtColor(current_frame, cv.COLOR_RGB2BGR)

    if detection_result_list:
        # print(detection_result_list)
        vis_image = visualize(current_frame, detection_result_list[0])
        cv.imshow('object_detector', vis_image)
        detection_result_list.clear()
    else:
        cv.imshow('object_detector', current_frame)

    # Stop the program if the ESC key is pressed.
    if cv.waitKey(1) == 27:
      break

detector.close()
cap.release()
cv.destroyAllWindows()







    


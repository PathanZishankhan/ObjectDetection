import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


baseoptions = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=baseoptions, score_threshold=0.5)
detector =  vision.ObjectDetector.create_from_options(options)

# img = cv.imread('image.png')
# imageRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

img = mp.Image.create_from_file('z.jpg')

detection_result = detector.detect(img)

print(detection_result)

image_copy = np.copy(img.numpy_view())


for detection in detection_result.detections:
    bbox = detection.bounding_box
    cv.rectangle(image_copy, (bbox.origin_x, bbox.origin_y), (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (0,255,0), 2)


    category = detection.categories[0]
    category_name = category.category_name
    probaScore = round(category.score, 2)
    txt = category_name + '(' + str(probaScore) + ')'
    cv.putText(image_copy, txt, (bbox.origin_x, bbox.origin_y + 20), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

rgbImage = cv.cvtColor(image_copy, cv.COLOR_BGR2RGB)
cv.imshow('result', rgbImage)
detector.close()
cv.waitKey(0)



    

    
import cv2
import numpy as np
from ultralytics import YOLO
import torch

#import pyrealsense2 as rs
#from realsense_depth import *
from time import time

# RESX = 1920
# RESY = 1080

# print("Torch version:",torch.__version__)
# print("Is CUDA enabled?",torch.cuda.is_available())
# # Load YOLOv5s model
# model = YOLO('yolov8x-pose.pt')
# model.to('cuda')

# cap = DepthCamera()


# Rewrite above code to be in separate functions
# Create a class that contains the functions
import depthEstimation as de


class PersonDetector:
    def __init__(self, cap, device='cpu'):
        global model
        model = YOLO('yolov8x-pose.pt')
        model.to(device)
        self.cap = cap
        self.depthToColorRes = (cap.depthRes[0] / cap.colorRes[0], cap.depthRes[1] / cap.colorRes[1])
        self.results = None
        self.frame = None
        self.depth_frame = None

    def update(self):
        ret, infrared_frame, depth_frame, frame = self.cap.get_frame()
        self.results = model(frame, conf=0.7, verbose=False, max_det=6, half=False)
        self.frame = frame
        self.depth_frame = depth_frame
    
    def getChestHeight(self, chestPoints):
        shoulder_midpoint = ((chestPoints[0][0] + chestPoints[1][0]) / 2, (chestPoints[0][1] + chestPoints[1][1]) / 2)
        waist_midpoint = ((chestPoints[2][0] + chestPoints[3][0]) / 2, (chestPoints[2][1] + chestPoints[3][1]) / 2)
        return np.sqrt((shoulder_midpoint[0] - waist_midpoint[0]) ** 2 + (shoulder_midpoint[1] - waist_midpoint[1]) ** 2)
        #return chest_bound[3] - chest_bound[1]
    
    def getDepth(self, chest_bound, depth_frame):
        distances = []
        for i in range(chest_bound[0], chest_bound[2]):
            for j in range(chest_bound[1], chest_bound[3]):
                distances.append(depth_frame[j][i])
        return np.median(distances) if len(distances) > 0 else 0
    
    def getChestBound(self, chest_points):
        xmin = min(chest_points[0][0], chest_points[1][0])
        xmax = max(chest_points[2][0], chest_points[3][0])
        ymin = min(chest_points[0][1], chest_points[1][1])
        ymax = max(chest_points[2][1], chest_points[3][1])
        chestBound = (int(xmin), int(ymin), int(xmax), int(ymax))
        return chestBound
    
    def getChestKeyPoints(self, result, threshold=0.6):
        keypoints = result.keypoints.data.cpu().numpy()
        chest_points = [keypoints[0][5], keypoints[0][6], keypoints[0][11], keypoints[0][12]]
        for chestPoint in chest_points:
            if chestPoint[2] < threshold:
                return None
        return chest_points
    
    # Depth Height ChestCenter = DHCT
    def getDHCPerTarget(self):
        sensorDepths = []
        sensorHeights = []
        chestCenters = []
        for result in self.results[0]:
            chestPoints = self.getChestKeyPoints(result)
            if chestPoints is not None:
                chestBound = self.getChestBound(chestPoints)
                chestCenter = (int((chestBound[0] + chestBound[2]) / 2), int((chestBound[1] + chestBound[3]) / 2))
                chestCenters.append(chestCenter)
                sensorDepths.append(self.getDepth(chestBound, self.depth_frame))
                sensorHeights.append(self.getChestHeight(chestPoints))
        return sensorDepths, sensorHeights, chestCenters
    
    def getDHCFrame(self, d, h, c, frame):
        annotated_frame = frame
        for i, (depth, height, center) in enumerate(zip(d, h, c)):
            if depth is not None and height is not None and center is not None:
                cv2.putText(annotated_frame, "Depth: {0:.2f} mm".format(depth), center, cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 100), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, "Height: {0:.2f} px".format(height), (center[0], center[1] + 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 200, 100), 2, cv2.LINE_AA)
                cv2.circle(annotated_frame, center, 5, (100, 200, 0), thickness=2, lineType=8, shift=0)
                cv2.putText(annotated_frame, "Target ID: {}".format(i), (center[0], center[1] + 80), cv2.FONT_HERSHEY_PLAIN, 3, (100, 0, 200), 2, cv2.LINE_AA)
        return annotated_frame
    
    def getFinalFrame(self, d, h, c, start_time):
        annotated_frame = self.getDHCFrame(d, h, c, self.frame)
        annotated_frame = self.calcFrameRate(annotated_frame, start_time)
        return annotated_frame
    
    def getTargetPositions(self, d, c):
        targetPositions = []
        for i, (depth, center) in enumerate(zip(d, c)):
            if depth is not None and center is not None:
                point3d = self.cap.get3d(center[0], center[1], depth)
                if point3d is not None:
                    targetPositions.append(point3d)
        return targetPositions
    
    def calcFrameRate(self, frame, start_time):
        current_time = time()
        elapsed_time = current_time - start_time
        frame_rate = 1 / elapsed_time
        cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 100), 2, cv2.LINE_AA)
        return frame




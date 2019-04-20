from __future__ import print_function

import numpy as np

from detect import *
from findObjects import Locator

import cv2
import multiprocessing as mp
import sys, os, threading

from segment import road_segmentation
from ipmdistance import getDistance
from navigate import safeZones

tf.reset_default_graph()
detection_object = Detector()

locate = Locator()

# def load_images(path):
#
# 	getFrames = []
# 	for imgs in sorted(os.listdir(path)):
# 		getFrames.append(Image.open(os.path.join(path, imgs)))
# 	return getFrames

# Get handle for frames
cap = cv2.VideoCapture('campusvideo_part3.mp4')
# Counter for frames
i=0

while(True):

	ret, frame = cap.read()
	H,W,C = frame.shape
	# pilImage = Image.fromarray(frame)
	orgFrame = frame.copy()
	print(i)

	if(i%10==0):
		frame, trackers, results = locate.runDetection(frame, detection_object)
		output = mp.Queue()
	else:
		frame, results = locate.parallelTracking(frame, trackers, output)

	if i==0:
		previousFrame = orgFrame
	if i>=1:
		frame, arrowDict = locate.motionVectors(results, previousFrame, frame)
		previousFrame = orgFrame

	output = mp.Queue()

	thread_distance = threading.Thread(target=getDistance, args=(frame, results, output,))
	thread_segment = threading.Thread(target=road_segmentation, args=(frame, output,))
	post = [thread_distance, thread_segment]

	for p in post:
		p.start()
	for p in post:
		p.join()

	res = [output.get() for p in post]
	for r in res:
		if type(r) == type([]):
			distance_to_objects, correlation, M, M_inv = r
		else:
			frame, road_mask = r

	if i>=1:
		safeZones(frame, arrowDict, distance_to_objects, correlation, road_mask, M, M_inv)

	font = cv2.FONT_HERSHEY_SIMPLEX
	for k in distance_to_objects.keys():
		midpoint_x = int(distance_to_objects[k][0]+distance_to_objects[k][2]/2)
		midpoint_y = int(distance_to_objects[k][1]+distance_to_objects[k][3])
		# cv2.line(frame, (midpoint_x, midpoint_y), (midpoint_x, H), (255,0,0), 5)
		cv2.putText(frame, str(k)[:4]+' meters',(midpoint_x,midpoint_y+3), font, 1, (255,0,0), 7, cv2.LINE_AA)


	# if i==0:
	# 	previousFrame = orgFrame
	# if i>=1:
	# 	frame = locate.motionVectors(results, previousFrame, frame)
	# 	previousFrame = orgFrame
	frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	cv2.imshow("Tracking", frame)
	i+=1
	k = cv2.waitKey(1) & 0xff
	if(k==27):
		break
	# cv2.imwrite("output/frames/frame_{}.png".format(i), frame)

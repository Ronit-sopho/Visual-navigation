from __future__ import print_function

import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rc
# from IPython.display import HTML

#
# plt.rcParams['figure.figsize'] = (15,12)
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

from detect import *
from utils import convert_to_original_size
from skimage.color import rgb2gray

import cv2
import multiprocessing as mp
import sys, os, threading

from segment import road_segmentation
from ipmdistance import getDistance

tf.reset_default_graph()
detection_object = Detector()


def load_images(path):
	getFrames = []
	for imgs in sorted(os.listdir(path)):
		getFrames.append(Image.open(os.path.join(path, imgs)))
	return getFrames

def SingleTracker(trackerObject, vid_frame, output):

	try:
		ret, bbox = trackerObject.update(vid_frame)
	except:
		pass
	if(ret):
		output.put(bbox)


path = 'test_images/data'

cap = cv2.VideoCapture('input_video.mp4')
i=0

while(True):

	ret, frame = cap.read()
	H,W,C = frame.shape
	pilImage = Image.fromarray(frame)
	print(i)
	if(i%10==0):
		pil_image, bounding_boxes = detection_object.infer(pilImage)
		# print(np.array(pil_image).shape)
		frame = np.asarray(pil_image)
		roi_list = []
		for cls, bboxes in bounding_boxes.items():
			for box, score in bboxes:
				if(np.all(box>0)):
					roi_list.append(box)

		ntrackers = len(roi_list)
		results = []
		trackers = []
		output = mp.Queue()
		for r in range(ntrackers):
			trck = cv2.TrackerMIL_create()
			x0,y0,x1,y1 = tuple(roi_list[r].astype(int))

			if x1-x0<=500 and y1-y0<=500:
				try:
					trck.init(frame, (x0,y0,x1-x0,y1-y0))
					trackers.append(trck)
					results.append((x0,y0,x1-x0,y1-y0))
				except:
					print("Error Encountered")
					continue

	else:

		threads = [threading.Thread(target=SingleTracker, args=(trck, frame, output,)) for trck in trackers]
		for t in threads:
			t.start()
		for t in threads:
			t.join()
		results = [output.get() for t in threads]
		for x in trackers:
			del x

		for nbox in results:
			p1 = (int(nbox[0]), int(nbox[1]))
			p2 = (int(nbox[0]+nbox[2]), int(nbox[1]+nbox[3]))
			cv2.rectangle(frame, p1, p2, (0,0,255), 2,1)
			# print([p1,p2])
		# frame = cv2_image

	# frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
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
		if type(r)==type({}):
			distance_to_objects=r
		else:
			frame=r
	# distance_to_objects = getDistance(frame, results)
	font = cv2.FONT_HERSHEY_SIMPLEX
	for k in distance_to_objects.keys():
		midpoint_x = int(distance_to_objects[k][0]+distance_to_objects[k][2]/2)
		midpoint_y = int(distance_to_objects[k][1]+distance_to_objects[k][3])
		cv2.line(frame, (midpoint_x, midpoint_y), (midpoint_x, H), (255,0,0), 5)
		cv2.putText(frame, str(k)[:4]+' meters',(midpoint_x,int((midpoint_y+H)/2)), font, 1, (255,0,0), 7, cv2.LINE_AA)

	# frame = road_segmentation(frame)
	cv2.imshow("Tracking", frame)
	i+=1
	k = cv2.waitKey(1) & 0xff
	if(k==27):
		break
	cv2.imwrite("output/frames/frame_{}.png".format(i), frame)

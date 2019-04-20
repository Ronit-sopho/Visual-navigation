from PIL import Image
import cv2
import numpy as np
import sys, os, threading

from skimage.feature import CENSURE
from skimage.color import rgb2gray
from motion import pyramid_lucas_kanade

class Locator(object):

    def runDetection(self, frame, detection_object):

        pilImage = Image.fromarray(frame)
        outputImage, bounding_boxes = detection_object.infer(pilImage)
        outputFrame = np.asarray(outputImage)

        detectedObjects = []
        trackers = []
        results = []

        for cls, bboxes in bounding_boxes.items():
            for box, score in bboxes:
                if np.all(box>0):
                    detectedObjects.append(box)

        ntrackers = len(detectedObjects)
        for r in range(ntrackers):
            trck = cv2.TrackerMIL_create()
            x0,y0,x1,y1 = tuple(detectedObjects[r].astype(int))
            if x1-x0<=500 and y1-y0<=500:
                try:
                    trck.init(frame, (x0,y0,x1-x0, y1-y0))
                    trackers.append(trck)
                    results.append((x0,y0,x1-x0, y1-y0))
                except:
                    print('Error Encountered')
                    continue

        return outputFrame, trackers, results


    def SingleTracker(self, trackerObject, vid_frame, output):

        try:
            ret, bbox = trackerObject.update(vid_frame)
        except:
            pass
        if(ret):
            output.put(bbox)


    def parallelTracking(self, frame, trackers, output):

        threads = [threading.Thread(target=self.SingleTracker, args=(trck, frame, output,)) for trck in trackers]
        for t in threads: t.start()
        for t in threads: t.join()

        results = [output.get() for t in threads]
        for t in trackers: del t

        for box in results:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0]+box[2]), int(box[1]+box[3]))
            cv2.rectangle(frame, p1, p2, (0,0,200), 2, 1)

        return frame, results

    def motionVectors(self, detectedObjects, currentFrame, nextFrame):

        censure = CENSURE()
        keypoints = np.array([]).reshape(-1,2)
        nkps = {}
        arrowDict = {}

        for num,region in enumerate(detectedObjects):
            x0,y0,w,h = region
            roi = rgb2gray(currentFrame[int(y0-5):int(y0+h+5), int(x0-5):int(x0+w+5)])
            try:
                censure.detect(roi)
                kps = censure.keypoints
                kps[:,1]+=int(x0)
                kps[:,0]+=int(y0)
                # kps = np.c_[kps,num*np.ones(kps.shape[0])]
                nkps[num] = kps.shape[0]
                keypoints = np.append(keypoints, kps, axis=0)
            except:
                print('Skipped ROI')
                return nextFrame, arrowDict

        # print(keypoints.shape)

        try:
            flow_vectors = pyramid_lucas_kanade(rgb2gray(currentFrame), rgb2gray(nextFrame), keypoints, window_size=9)
        except:
            return nextFrame, arrowDict

        counter = 0
        aggregate_vectors = np.hstack((keypoints, flow_vectors))

        for k in nkps.keys():
            if nkps[k] != 0:
                vec = np.sum(aggregate_vectors[counter:counter+nkps[k],:], axis=0)
                avgY = vec[0]/nkps[k]
                avgX = vec[1]/nkps[k]
                p1 = (int(avgX), int(avgY))
                p2 = (int(avgX+vec[3]), int(avgY+vec[2]))
                arrowDict[k] = (p1,(int(vec[3]), int(vec[2])))
                cv2.arrowedLine(nextFrame, p1, p2, (225,32,33), 3)
                counter+=nkps[k]

        return nextFrame, arrowDict

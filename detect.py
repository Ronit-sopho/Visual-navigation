# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time
import os

import yolo_v3
# import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image


if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

class Detector(object):

	def __init__(self):

	    config = tf.ConfigProto()
	    config.gpu_options.allow_growth = True
	    config.gpu_options.per_process_gpu_memory_fraction = 0.5

	    self.classes = load_coco_names('coco.names')

	    self.model = yolo_v3.yolo_v3
	    self.boxes, self.inputs = get_boxes_and_inputs(self.model, len(self.classes), 416, 'NHWC')
	    self.saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

	    self.sess = tf.Session(config=config)
	    t0 = time.time()
	    self.saver.restore(self.sess, './saved_model/model.ckpt')
	    print('Model restored in {:.3f}s'.format(time.time()-t0))


	def infer(self, input_image):

	    # img = Image.open('test_images/car2.png')
	    img = input_image.copy()
	    img_resized = letter_box_image(img, 416, 416, 128)
	    img_resized = img_resized.astype(np.float32)


	    t0 = time.time()
	    detected_boxes = self.sess.run(self.boxes, feed_dict={self.inputs: [img_resized]})

	    filtered_boxes = non_max_suppression(detected_boxes,
	                                         confidence_threshold=0.8,
	                                         iou_threshold=0.5)
	    # print(filtered_boxes)
	    print("Predictions found in {:.3f}s".format(time.time() - t0))

	    draw_boxes(filtered_boxes, img, self.classes, (416, 416), True)
	    # img.save('out.png')
	    return img,filtered_boxes

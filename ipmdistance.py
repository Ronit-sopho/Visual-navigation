import cv2
import numpy as np

def getDistance(input_image, object_boxes, output):

    # Get a bird's eye view of the image
    # H = height
    # W = width
    # C = channels
    H,W,C = input_image.shape

    # define the image slicing parameter (ROI)
    s = 430 # works best for our images

    # define the perspective points and their mapping
    source  = np.float32([[0,H],[W,H],[0,s],[W,s]])
    destination = np.float32([[550,H],[800,H],[0,0],[W,0]])

    # Get the transformation matrix
    M = cv2.getPerspectiveTransform(source, destination)
    M_inv = cv2.getPerspectiveTransform(destination, source)

    # get the transformed image
    warped_img = cv2.warpPerspective(input_image.copy(), M, (W,H))
    # write intermdiate ouput
    cv2.imwrite('output/ipm_image.jpg', warped_img)

    # dictionary containing distances
    distance = {}

    for box in object_boxes:
        top_left = (int(box[0]), int(box[1]), 1)
        bottom_right = (int(box[0]+box[2]), int(box[1]+box[3]), 1)
        # print(bottom_right)
        # bottom_left = (int(box[0]), int(box[1]+box[3]))
        if bottom_right[1]>s:
            skewed_box = np.matmul(M, np.array(bottom_right))
            skewed_box = skewed_box/skewed_box[2]
            dist = ((H - skewed_box[1])/130)*3
            distance[dist] = box

    # return distance
    output.put(distance)

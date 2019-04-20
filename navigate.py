import cv2
import numpy as np

def safeZones(frame, arrowDict, distances, correspondence, road_mask, M, M_inv):

    # Get direction of object
    set1 = set(correspondence.keys())
    set2 = set(arrowDict.keys())

    common = set1 & set2
    directions = {}
    for k in common:

        if arrowDict[k][1][1]>=0:
            directions[k] = "B"
        else:
            directions[k] = "F"

        if arrowDict[k][1][0]>=0:
            directions[k]+="R"
        else:
            directions[k]+="L"

    for k in directions.keys():

        x,y,w,h = distances[correspondence[k]]
        bottom_right = np.array([x+w,y+h,1])
        ipm_coord_br = np.matmul(M,bottom_right)
        ipm_coord_br = ipm_coord_br/ipm_coord_br[2]
        # coord_mat = np.array([[x,y,1],[x+w,y,1],[x,y+h,1],[x+w,y+h,1]])

        # skewed_box = np.matmul(M,coord_mat.T)
        # skewed_box = skewed_box[:,:]/skewed_box[2,:]
        skewed_box = np.array([ipm_coord_br,ipm_coord_br,ipm_coord_br,ipm_coord_br]).T
        extend = np.array([[50,-100,-100,50],[50,50,-100,-100],[0,0,0,0]])
        extended_box = skewed_box+extend

        skew_inv = (np.matmul(M_inv, extended_box))
        skew_inv = skew_inv/skew_inv[2,:]
        skew_inv = skew_inv.T
        skew_inv = skew_inv[:,:2]
        skew_inv = np.ceil(skew_inv)
        skew_inv = np.array([skew_inv], dtype=np.int32)
        cv2.fillPoly(frame, skew_inv, (0,0,255))

        # bottom_left = np.array([x,y+h,1])
        # bottom_right = np.array([x+w,y+h,1])
        # ipm_coord_br = np.matmul(M,bottom_right)
        # ipm_coord_br = ipm_coord_br/ipm_coord_br[2]
        # ipm_coord_bl = np.matmul(M,bottom_left)
        # ipm_coord_bl = ipm_coord_bl/ipm_coord_bl[2]
        # ipm_w = ipm_coord_br[0]-ipm_coord_bl[0]
        # ipm_h = ipm_w
        #
        # if directions[k][1]=="R":
        #     ipm_coord_br[1]+=90
        #     ipm_coord_br[0]+=50
        #
        #
        # else:
        #     ipm_coord = np.matmul(M,bottom_left)
        #     ipm_coord = ipm_coord/ipm_coord[2]
        #     ipm_coord[1]+=90
        #     ipm_coord[0]-=50
        #

import cv2
import numpy as np
import matplotlib.image as mpimg

def detectLanes(road_mask, M, input_image_new):

    # Get final_filled
    # final_filled = mpimg.imread('final_filled_153.jpg')
    final_filled = road_mask
    # print(final_filled.shape)
    # final_filled = cv2.cvtColor(final_filled, cv2.COLOR_BGR2GRAY)
    input_img = input_image_new.copy()
    #Get input images

    # undist = mpimg.imread('frame0153.jpg')
    undist = input_image_new.copy()

    #print(final_filled.shape)
    ##mask = cv2.bitwise_and(image,image, mask=final_filled)
    #mask_orrg = cv2.bitwise_and(input_img, input_img, mask = final_filled)
    # input_img = mpimg.imread('frame0153.jpg')
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    jmd= cv2.bitwise_and(gray,final_filled)
    #plt.imshow(jmd, cmap='gray')
    #img = cv2.imread('frame0378.jpg')
    H,W = jmd.shape
    # gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    s = 625   # slicing parameter

    src = np.float32([[0,H],[W,H],[0,s],[W,s]])
    dst = np.float32([[600,H],[1000,H],[0,0],[W,0]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # cropped_img = jmd[s:,:]
    #print(np.sum(gray))
    warped_img = cv2.warpPerspective(jmd, M, (W,H))

    # GEt warped image

    # cv2.imwrite('warp_out_153.jpg', warped_img)
    #plt.imshow(warped_img, cmap='gray')
    #print(np.sum(warped_img))
    b = np.matmul(M, np.array([960,1020,1]))
    b = b/b[2] #vehicle centre location
    #print(warped_img[100,800])
    ret,thresh2 = cv2.threshold(warped_img,200,255,cv2.THRESH_BINARY)
    # cv2.imwrite('thresh2_0153.jpg', thresh2)
    histogram = np.sum(thresh2[:,:], axis=0)
    # cv2.imwrite('histogram.jpg',histogram)
    out_img = (np.dstack((thresh2,thresh2,thresh2)))
    leftx_base = np.argmax(histogram[:500])
    midx_base = np.argmax(histogram[500:800]) + 500
    rightx_base = np.argmax(histogram[800:]) + 800
    peaks = [leftx_base, midx_base, rightx_base]
    nwindows = 9
    window_height = np.int(thresh2.shape[0]/nwindows)
    nonzero = thresh2.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 20
        # Set minimum number of pixels found to recenter window
    minpix = 50
        # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    lanes = []
    warp_zero = np.zeros_like(thresh2).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    lanes = []
    for n in range(np.size(peaks)):
        leftx_current = peaks[n]
        left_lane_inds = []
        for window in range(nwindows):
              # Identify window boundaries in x and y (and right and left)
            win_y_low = int(thresh2.shape[0] - (window+1)*window_height)
            win_y_high = int(thresh2.shape[0] - window*window_height)
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3)
            #plt.imshow(out_img)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        # cv2.imwrite('fjslfjsf.jpg',out_img)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        #print(left_fit)
        ploty = np.linspace(0, thresh2.shape[0]-1, thresh2.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.transpose(np.vstack([left_fitx+20, ploty]))])
        pts_right = np.flip(pts_right, axis=0)
        pts = np.hstack((pts_left, pts_right))
        #print(pts_left)
        pts = np.int_([pts])
        # cv2.fillPoly(color_warp, pts, (24,60,242))
        # cv2.imwrite('boxes_on_lanes.jpg', out_img)
        lanes.append(pts_left)

    if(b[0]>midx_base):
      x = np.array(lanes[1][0], dtype=object)
      y = np.array(lanes[2][0], dtype=object)
    else:
      x = np.array(lanes[0][0], dtype=object)
      y = np.array(lanes[1][0], dtype=object)

    y = np.flip(y, axis=0)
    xy = np.vstack((x,y))
    #xy = np.int(xy)
    xy = xy.astype('int')
    #print(xy)
    cv2.fillPoly(color_warp, [xy], (242,60,24))
    # cv2.imwrite('color_warp_153.jpg',color_warp)
    #plt.imshow(color_warp)
    undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (thresh2.shape[1], thresh2.shape[0]))
        # overlay
        #newwarp = cv.cvtColor(newwarp, cv.COLOR_BGR2RGB)
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
    # plt.imshow(result)
    # cv2.imwrite('lanes_153.jpg',result)
    return result

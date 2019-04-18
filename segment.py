import cv2
import numpy as np

def road_segmentation(input_image, output):

    # Create a copy of the image and reorder its array to RGB
    inp_img = input_image.copy()
    inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
    inp_img = np.double(inp_img)

    # Perform illumination invariance on the Image
    alpha = 0.05
    ii_image = 0.5 + np.log(inp_img[:,:,1]) - alpha*np.log(inp_img[:,:,2]) - (1-alpha)*np.log(inp_img[:,:,0])
    ii_image = 255*ii_image
    # Write the intermdiate output
    cv2.imwrite('output/ii_image.jpg', ii_image)

    # Perform Road Segmentation by thresholding and masking the illuminaiton invariant Image
    ret, thresh = cv2.threshold(ii_image, 120, 150, cv2.THRESH_BINARY)
    final_masked = np.uint8(thresh)
    height = final_masked.shape[0]
    width = final_masked.shape[1]

    # Erode the image and find contours to remove false positives
    kernel = np.ones((3,3), np.uint8)
    final_eroded = cv2.erode(final_masked, kernel, iterations=1)
    contours,_ = cv2.findContours(final_eroded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_masked = cv2.drawContours(final_eroded, contours, -1, 0, 3)

    # Refine the mask with (tophat image = input image - opening image("remove false positives"))
    final_waste = cv2.morphologyEx(final_masked,cv2.MORPH_TOPHAT,kernel, iterations = 2)
    final_waste = cv2.bitwise_not(final_waste)
    final_masked = cv2.bitwise_and(final_waste,final_masked)
    # final_masked = cv2.line(final_masked,(40,height),(400,height),255,100)

    # Fill the holes in the mask representing the road
    final_flood = final_masked.copy()
    h, w = final_masked.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(final_flood,mask,(0,0),255)
    final_flood = cv2.bitwise_not(final_flood)
    final_filled= cv2.bitwise_or(final_masked,final_flood)
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (255,0,0)

    mask = cv2.bitwise_and(image, image, mask = final_filled)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    # Write intermediate output
    # cv2.imwrite('mask.jpg',mask)

    gray = cv2.cvtColor(input_image.copy(), cv2.COLOR_BGR2GRAY)
    jmd = cv2.bitwise_and(gray, final_filled)
    undist = input_image.copy()
    result = cv2.addWeighted(undist, 1, mask, 0.3, 0)

    # Write the output of segmentation for each frame
    cv2.imwrite('output/res.jpg', result)

    # return result
    output.put(result)

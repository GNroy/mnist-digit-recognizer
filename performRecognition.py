import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import sys

# Load the classifier and filter
clf = joblib.load("digits_cls.pkl")
filterclf = joblib.load("filter_cls.pkl")

# Read the input image
im = cv2.imread(sys.argv[1])

# Hardcoded parameters to detect red digits
lower = np.array((15,15,140), dtype = "uint8")
upper = np.array((110,140,255), dtype = "uint8")

# Threshold the image
im_th = cv2.inRange(im, lower, upper)
im_th = cv2.GaussianBlur(im_th, (5, 5), 0)

# Find contours in the image
_, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict the digit
for rect in rects: 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    # Noise dropping No1: Throw out too small and too wide rectangles
    if pt1 >= 0 and pt2 >= 0 and leng >= 30 and rect[2]<=rect[3]*1.4 and rect[3]<=rect[2]*4:
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        # Noise dropping No2:
        # Check filter's answer and whether the probability of the right answer is good enough
        if int(filterclf.predict(np.array([roi_hog_fd], 'float64'))[0]) == 1 and np.max(clf.predict_proba(np.array([roi_hog_fd], 'float64'))) > 0.2:
            # Draw the rectangles
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
            cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1] + rect[3]),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

# Show the resulting image
cv2.imshow("Resulting Image", im)
cv2.waitKey()
cv2.destroyAllWindows()




import cv2

# https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md#orb-oriented-fast-and-rotated-brief
# detector = cv2.ORB_create(
#     edgeThreshold=15,  
#     patchSize=31, 
#     nlevels=8, 
#     fastThreshold=0, #20 
#     scaleFactor=1.2, 
#     WTA_K=2,
#     scoreType=cv2.ORB_HARRIS_SCORE, 
#     firstLevel=0, 
#     nfeatures=1000
# )

# try sift instead, or anything scale invariant
detector = cv2.SIFT_create()
# that, but supplying all named parameters
detector = cv2.SIFT_create(
    nfeatures=2000,
    nOctaveLayers=6,
    contrastThreshold=0.04,
    edgeThreshold=10,
    sigma=1.6 # 	The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
)

# FLANN_INDEX_LSH = 6
# flann_params = dict(
#     algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=2
# )
# matcher = cv2.FlannBasedMatcher(flann_params, {})
matcher = cv2.BFMatcher(
    normType=cv2.NORM_L2, crossCheck=False
)


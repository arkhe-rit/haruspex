import cv2
import numpy as np
from time import perf_counter


# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# represents a pipeline of functions to be applied to a frame
class Pipeline():
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, *args):
        if len(self.funcs) == 0:
            return args[0]
            
        result = self.funcs[0](*args)
        for func in self.funcs[1:]:
            result = func(result)
        return result
    
    def map(self, f):
        return Pipeline(*(self.funcs + (f,)))
    
    def contramap(self, f):
        return Pipeline(*(f,) + self.funcs)


def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def threshold(frame):
    return cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5)

def background_threshold(params):
    def f(frame):
        blockSize = params['threshold_blockSize']
        blockSize = blockSize if blockSize % 2 == 1 else blockSize + 1
        C = params['threshold_C']

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)

        # if np.mean(gray) > 100:
        #     gray = cv2.bitwise_not(gray)

        img_w, img_h = np.shape(frame)[:2]

        # find the most common level in the image (background level)
        # hist = cv2.calcHist(images=[blur], channels=[0], mask=None, histSize=[32], ranges=[0, 256])
        # bkg_level = np.argmax(hist) * 256 / 32
        bkg_level_top = blur[int(img_h/100)][int(img_w/2)]
        bkg_level = bkg_level_top

        thresh_level = bkg_level + 30 # BKG_THRESH
        # if background is too bright, invert the image
        # if bkg_level > 100:
        #     blur = cv2.bitwise_not(blur)
        #     thresh_level = 255 - thresh_level

        # print thresh_level
        print(thresh_level)
        thresholdType = cv2.THRESH_BINARY_INV

        # retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)

        # retval, thresh = cv2.threshold(src=blur, thresh=thresh_level, maxval=255, type=thresholdType)
        # do gauusian adaptive thresholding with named parameters
        thresh = cv2.adaptiveThreshold(
            src=blur, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=thresholdType, blockSize=blockSize, C=C
        )
        
        blurred_thresh = cv2.GaussianBlur(thresh,(21,21),0)

        return blurred_thresh
    return f

def otsu_threshold(params):
    def f(frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh
    return f

def contours(output_out):
    def f(frame):
        contours, hierarchy = cv2.findContours(
            image=frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )

        def isRectangular(contour):
            perimeter = cv2.arcLength(contour, True) 
            approx = cv2.approxPolyDP(
                curve=contour, epsilon=0.1 * perimeter, closed=True
            )
            return len(approx) == 4
        
        def ratioIsNotSeriouslyNarrow(contour):
            # x, y, w, h = cv2.boundingRect(contour)
            # aspect_ratio = max(w, h) / min(w, h)
            perimeter = cv2.arcLength(contour, True) 
            approx = cv2.approxPolyDP(
                curve=contour, epsilon=0.1 * perimeter, closed=True
            )
            # side lengths
            a, b, c, d = [np.linalg.norm(approx[i] - approx[i-1]) for i in range(4)]
            max_side = max(a, b, c, d)
            min_side = min(a, b, c, d)
            aspect_ratio = max_side / min_side

            is_convex = cv2.isContourConvex(approx)

            return is_convex and 1 < aspect_ratio < 3
        
        def areaIsSizedJustRight(contour):
            # return 200 ** 2 < cv2.contourArea(contour) < 500 ** 2
            return 200 ** 2 < cv2.contourArea(contour) < 800 ** 2
        
        def contourHasNoParent(contour_i):
            return hierarchy[0][contour_i][3] == -1
        
        def filter_similar_contours(contours, threshold=0.6):
            def IoU(rect1, rect2):
                x1, y1, w1, h1 = rect1
                x2, y2, w2, h2 = rect2
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)

                if x_right < x_left or y_bottom < y_top:
                    return 0.0

                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                union_area = w1 * h1 + w2 * h2 - intersection_area

                return intersection_area / union_area

            rects = [cv2.boundingRect(c) for c in contours]
            rects_areas = [w * h for x, y, w, h in rects]
            keep = [True] * len(contours)

            for i in range(len(contours)):
                if not keep[i]:
                    continue

                for j in range(i + 1, len(contours)):
                    if IoU(rects[i], rects[j]) > threshold:
                        to_ditch = i if rects_areas[i] < rects_areas[j] else j
                        keep[to_ditch] = False

            return [contours[i] for i in range(len(contours)) if keep[i]]

        # filtered_contours = all contours that are rectangular, are not too narrow, and have larger areas
        filtered_contours_with_i = list(filter(
            lambda ic: isRectangular(ic[1]) and 
                       ratioIsNotSeriouslyNarrow(ic[1]) and
                       areaIsSizedJustRight(ic[1]),# and 
                    #    contourHasNoParent(ic[0]), 
            enumerate(contours)
        ))
        filtered_contours = [c for i, c in filtered_contours_with_i]

        filtered_contours = filter_similar_contours(filtered_contours)

        # colored_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # for each contour, draw approximated polygon
        # approximations = [cv2.approxPolyDP(
        #     curve=contour, epsilon=0.1 * cv2.arcLength(contour, True), closed=True
        # ) for contour in filtered_contours]
        # cv2.drawContours(image=colored_frame, contours=approximations, contourIdx=-1, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # cv2.drawContours(image=colored_frame, contours=filtered_contours, contourIdx=-1, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        
        output_out['contours'] = filtered_contours
        return frame
    return f

def isolate_title_area(image):
    height, width = image.shape
    # title_area is bottom 15% and center 90% of image
    title_area = image[int(height * 0.85):, int(width * 0.05):int(width * 0.95)]

    return title_area

def sharpen(image):
    # gaussian blur
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    # unsharp mask
    sharpened = cv2.addWeighted(image, 2.0, blurred, -1.0, 0)
    return sharpened
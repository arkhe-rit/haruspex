
# python standard imports
from abc import ABC, abstractmethod
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import os

# 3rd party imports
import cv2
import numpy as np
import pytesseract
from Levenshtein import distance as lev_distance

# local imports
import pipeline
from pipeline import Pipeline, perspective_transform, make_square
from tarot_cards import cards_keypoints
from threaded_camera import ThreadedCamera
from publish import publish, subscribe, subscribe_once

pytesseract.pytesseract.tesseract_cmd = fr'{os.getcwd()}\tesseract\tesseract.exe'

FPS = 15

params = {
    'threshold_blockSize': 71, #27, # 591
    'threshold_C': 15, #15 #40,
    'show_threshold': False,
    'debug_show': False,
    'roi_top': 0,
    'roi_left': 0,
    'roi_size_factor': 1,
    'show_roi': False,
    'rotate_angle': 0
}
# At home
params['threshold_blockSize'] = 234
params['threshold_C'] = 12

cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)

cv2.createTrackbar('Threshold block size', 'Controls', params['threshold_blockSize'], 1000, lambda val: params.update({'threshold_blockSize': val}))
cv2.setTrackbarMin('Threshold block size', 'Controls', 3)
cv2.createTrackbar('Threshold C', 'Controls', params['threshold_C'], 100, lambda val: params.update({'threshold_C': val}))
cv2.setTrackbarMin('Threshold C', 'Controls', -10)
cv2.createTrackbar('Show threshold', 'Controls', 0, 1, lambda val: params.update({'show_threshold': True if val > 0.5 else False}))
cv2.createTrackbar('Debug show', 'Controls', 1 if params['debug_show'] else 0, 1, lambda val: params.update({'debug_show': True if val > 0.5 else False}))
# Create four trackbars for edges of region of interest
cv2.createTrackbar('ROI top', 'Controls', 0, 100, lambda val: params.update({'roi_top': val / 100}))
cv2.createTrackbar('ROI left', 'Controls', 0, 100, lambda val: params.update({'roi_left': val / 100}))
cv2.createTrackbar('ROI size factor', 'Controls', 100, 100, lambda val: params.update({'roi_size_factor': val / 100}))
# show roi
cv2.createTrackbar('Show ROI', 'Controls', 1 if params['show_roi'] else 0, 1, lambda val: params.update({'show_roi': True if val > 0.5 else False}))
# a trackbar for rotation, allowing only 0, 90, 180, 270
cv2.createTrackbar('Rotate angle', 'Controls', params['rotate_angle'], 270, lambda val: params.update({'rotate_angle': val // 90 * 90}))


def find_best_text_match(cam_image):
    # determine if cam_image is vertical or horizontal
    is_vertical = cam_image.shape[0] > cam_image.shape[1]

    cam_image = make_square(cam_image)
    cam_image = pipeline.to_grayscale(cam_image)

    cam_rotated_titles = []
    if is_vertical:
        cam_rotated_titles = [
            pipeline.isolate_title_area(cam_image),
            pipeline.isolate_title_area(cv2.rotate(cam_image, cv2.ROTATE_180))
        ]
    else:
        cam_rotated_titles = [
            pipeline.isolate_title_area(cv2.rotate(cam_image, cv2.ROTATE_90_CLOCKWISE)),
            pipeline.isolate_title_area(cv2.rotate(cam_image, cv2.ROTATE_90_COUNTERCLOCKWISE))
        ]

    # invert colors of cam_rotated_titles
    cam_rotated_titles = [cv2.bitwise_not(img) for img in cam_rotated_titles]
    
    # sharpen cam_rotated_titles
    # cam_rotated_titles = [pipeline.sharpen(img) for img in cam_rotated_titles]
    # cam_rotated_titles = [pipeline.otsu_threshold(img) for img in cam_rotated_titles]
    cam_rotated_titles = [pipeline.sharpen(img) for img in cam_rotated_titles]

    def ocr_psm11(img):
        return pytesseract.image_to_string(img, config=r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c tessedit_do_invert=0')

    def ocr_psm7(img):
        return pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c tessedit_do_invert=0')

    def ocr_psm13(img):
        return pytesseract.image_to_string(img, config=r'--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c tessedit_do_invert=0')


    if params['debug_show']:
        for idx, img in enumerate(cam_rotated_titles):
            print("---")
            print("PSM11", ocr_psm11(img))
            print("PSM7", ocr_psm7(img))
            print("PSM13", ocr_psm13(img))
            print("---")
            cv2.imshow('cam_rotated_titles', img)
            cv2.waitKey(0)

    # smallest dimension of cam_rotated_titles[0]
    smallest_dim = min(cam_rotated_titles[0].shape[0], cam_rotated_titles[0].shape[1])
    # resize cam_rotated_titles to by 30 / smallest_dim
    resize_factor = 30 / smallest_dim
    cam_rotated_titles = [cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor) for img in cam_rotated_titles]

    with ThreadPoolExecutor() as executor:
        psm11_calls = [(img, ocr_psm11) for img in cam_rotated_titles]
        psm7_calls = [(img, ocr_psm7) for img in cam_rotated_titles]
        psm13_calls = [(img, ocr_psm13) for img in cam_rotated_titles]

        cam_rotated_text = list(
            executor.map(lambda args: args[1](args[0]), 
                         psm11_calls + psm7_calls + psm13_calls)
        )
    # cam_rotated_text = ["death" for _ in cam_rotated_titles]
    # trim everything in cam_rotated_text & remove empty strings
    cam_rotated_text = [text.strip() for text in cam_rotated_text if len(text.strip()) > 1]

    # distance to each card in cards_keypoints
    def distance_to_card(card_name):
        ref_image = cards_keypoints[card_name]['image']
        card_text = cards_keypoints[card_name]['text']
        card_text_short = cards_keypoints[card_name]['text_short']

        # for each cam_rotated_text, check if actually just contains card_text
        cam_contains_card_text = [card_text_short in cam_text for cam_text in cam_rotated_text]
        if any(cam_contains_card_text):
            return 0

        cam_text_distances = [lev_distance(card_text, cam_text) for cam_text in cam_rotated_text]
        cam_text_distances = [dist / len(card_text) for dist in cam_text_distances]
        min_cam_text_distance = min(cam_text_distances, default=100000)
        return min_cam_text_distance
    
    distances = [(card_name, distance_to_card(card_name)) for card_name in cards_keypoints.keys()]
    distances.sort(key=lambda x: x[1])
    closest_cards = [card_name for card_name, _ in distances[:5]]
    return closest_cards

    # return best_match

def find_best_template_match(cam_image):
    cam_image = make_square(cam_image)
    cam_image = pipeline.to_grayscale(cam_image)

    best_match = None
    best_match_quality = 0

    for idx, tarot_card in enumerate(cards_keypoints):
        ref_image = cards_keypoints[tarot_card]['image']

        if ref_image.shape != cam_image.shape:
            cam_image = cv2.resize(cam_image, ref_image.shape)            

        # Perform template matching
        result = cv2.matchTemplate(cam_image, ref_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        match_quality = max_val * 100

        if params['debug_show']:
            # show ref_image and cam_image side by side
            ref_image_small = cv2.resize(ref_image, (ref_image.shape[1] // 2, ref_image.shape[0] // 2))
            cam_image_small = cv2.resize(cam_image, (cam_image.shape[1] // 2, cam_image.shape[0] // 2))
            matches_img = np.concatenate((ref_image_small, cam_image_small), axis=1)
            matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(matches_img, f'{match_quality}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Matches', matches_img)
            cv2.waitKey(0)

        if match_quality > best_match_quality:
            best_match = tarot_card
            best_match_quality = match_quality

    return best_match

def control(params_out):
    key = cv2.waitKey(1)
    if key == ord('q'):
        return False
    
    # every 10 frames, print the parameters
    if cv2.getTickCount() % 10 == 0:
        print("Params ", params_out)

    return True

def horizontal_concat(images, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in images)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in images]
    return cv2.hconcat(im_list_resize)

def vertical_concat(images, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in images)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in images]
    return cv2.vconcat(im_list_resize)

def main():
    vid = ThreadedCamera(2, cv2.CAP_DSHOW)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    output = {}
        #         .map(pipeline.background_threshold(params)) \
        
    transform_stream = Pipeline() \
        .map(pipeline.rotate(params)) \
        .map(pipeline.narrow_to_roi(params))
    
    process_stream = Pipeline() \
        .map(pipeline.otsu_threshold) \
        .map(pipeline.contours(output))
    # process_stream = Pipeline() \
    #     .map(pipeline.background_threshold(params)) \
    #     .map(pipeline.contours(output))  


    should_continue = True

    try:
        while should_continue:
            time.sleep(1 / FPS)

            raw_frame = vid.latest()
            transformed_frame = transform_stream(raw_frame)
            processed_frame = process_stream(transformed_frame)
            frame = processed_frame

            # if show_roi
            if params['show_roi']:
                cv2.imshow('ROI', cv2.resize(pipeline.narrow_to_roi(params)(frame), (720, 405)))

            contours = output['contours']
            # sort by center x
            contours = sorted(contours, key=lambda contour: np.mean(contour[:, 0, 0]))

            frame_to_show = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR) if params['show_threshold'] else transformed_frame
            approximations = [
                cv2.approxPolyDP(curve=contour, epsilon=0.1 * cv2.arcLength(contour, True), closed=True) 
                for contour in contours
            ]

            shouldContinue = control(params)  
            if not shouldContinue:
                break   

            # compare the frame
            # for each contour, find its bounding box
            boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
            
            matches = []
            highDefFrame = pipeline.narrow_to_roi(params)(raw_frame)

            # for each approx
            for approx in approximations:
                # show image

                highDefImage = perspective_transform(highDefFrame, approx)
                cv2.imshow('highDefImage', highDefImage)
                # find the best match
                t0 = time.time()
                result = find_best_text_match(highDefImage)
                t1 = time.time()
                ms = (t1 - t0) * 1000
                print(f'find_best_text_match took {ms} milliseconds')

                matches.append(result[0])
                
            cv2.drawContours(image=frame_to_show, contours=approximations, contourIdx=-1, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            # for each approximation i, draw the text matches[i] just below bounding box
            for i, (x, y, w, h) in enumerate(boundingBoxes):
                cv2.putText(frame_to_show, matches[i], (x + 50, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
                              
            cv2.imshow("Found Cards", cv2.resize(frame_to_show, (720, 405)))

            print(matches)
            if len(matches) > 0:
                publish(channel='haruspex-cards-observed', 
                        message=matches)
            #time the following
            # t0 = None
            # t1 = None
            # def cont(message):
            #     t1 = time.time()
            #     ms = (t1 - t0) * 1000
            #     print(f'publish roundtrip took {ms} milliseconds')
            #     print(message)

            # subscribe_once(channel="haruspex-cards-observed-ack", callback=cont)
            # t0 = time.time()
            # result = publish(channel='haruspex-cards-observed', 
            #                  message=['death', 'hanged man', 'temperance'])

    except Exception as e:
        print("Error:", e)
        print(traceback.format_exc())

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

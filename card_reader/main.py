# Code rules:
# - All OpenCV functions should be called with named parameters

from abc import ABC, abstractmethod
import cv2
import numpy as np
import time
import traceback
import pytesseract
import os
from Levenshtein import distance as lev_distance
from concurrent.futures import ThreadPoolExecutor

import pipeline
from pipeline import Pipeline
from camera_capture import oak_capture_generator, video_capture_generator
from video_capture import VideoCapture

from tarot_cards import cards_keypoints
from feature_detector import detector, matcher
from four_point_transform import four_point_transform, perspective_transform, make_square
from threaded_camera import ThreadedCamera

pytesseract.pytesseract.tesseract_cmd = fr'{os.getcwd()}\tesseract\tesseract.exe'

FPS = 15

params = {
    'threshold_blockSize': 71, #27, # 591
    'threshold_C': 15, #15 #40,
    'show_threshold': False,
    'debug_show': False,
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

    # cam_rotated_titles = [pipeline.isolate_title_area(img) for img in [
    #     cam_image,
    #     cv2.rotate(cam_image, cv2.ROTATE_90_CLOCKWISE),
    #     cv2.rotate(cam_image, cv2.ROTATE_180),
    #     cv2.rotate(cam_image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    # ]]
    # invert colors of cam_rotated_titles
    cam_rotated_titles = [cv2.bitwise_not(img) for img in cam_rotated_titles]
    
    # sharpen cam_rotated_titles
    cam_rotated_titles = [pipeline.sharpen(img) for img in cam_rotated_titles]

    def ocr_psm11(img):
        return pytesseract.image_to_string(img, config=r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c tessedit_do_invert=0')

    def ocr_psm7(img):
        return pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c tessedit_do_invert=0')

    if params['debug_show']:
        for idx, img in enumerate(cam_rotated_titles):
            print("---")
            # print(pytesseract.image_to_string(img, config=r'--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            print(ocr_psm11(img))
            print(ocr_psm7(img))
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

        cam_rotated_text = list(
            executor.map(lambda args: args[1](args[0]), 
                         psm11_calls + psm7_calls)
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

def find_best_match(cam_image):
    cam_image = make_square(cam_image)
    cam_image = pipeline.to_grayscale(cam_image)
    cam_rotated_titles = [pipeline.isolate_title_area(img) for img in [
        cam_image,
        cv2.rotate(cam_image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(cam_image, cv2.ROTATE_180),
        cv2.rotate(cam_image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]]

    # for idx, cam_rotation in enumerate(cam_rotated_titles):
    #     cv2.imshow('cam_title_area', cam_rotation)
    #     cv2.waitKey(0)

    # edges = cv2.Canny(cam_image,100,200)

    # cv2.imshow('cam_image', cam_image)
    # cv2.waitKey(0)

    kp_cam, des_cam = detector.detectAndCompute(image=cam_image, mask=None)

    # show the keypoints on the cam_image and wait for keypress
    # cam_image = cv2.drawKeypoints(image=cam_image, keypoints=kp_cam, 
    #                   outImage=cam_image, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # cv2.imshow('Keypoints', cam_image)
    # cv2.waitKey(0)

    best_match = None
    best_match_quality = 0
    for idx, tarot_card in enumerate(cards_keypoints):
        ref_image = cards_keypoints[tarot_card]['image']
        # ref_image = cv2.Canny(ref_image,100,200)

        # is canny a good idea or not

        ref_title = pipeline.isolate_title_area(ref_image)

        kp_tarot = cards_keypoints[tarot_card]['keypoints']
        des_tarot = cards_keypoints[tarot_card]['descriptors']        
        matches_full_card = matcher.knnMatch(des_cam, des_tarot, k=2)

        kp_ref_title = cards_keypoints[tarot_card]['keypoints_title']
        des_ref_title = cards_keypoints[tarot_card]['descriptors_title']
        matches_titles = []
        for cam_title_area in cam_rotated_titles:
            kp_cam_title, des_cam_title = detector.detectAndCompute(image=cam_title_area, mask=None)
            these_title_matches = matcher.knnMatch(des_cam_title, des_ref_title, k=2)
            matches_titles.append(these_title_matches)

            if params['debug_show']:
                matches_to_show = [m for m, n in these_title_matches if m.distance < 0.95 * n.distance]
                # sort by quality
                matches_to_show.sort(key=lambda m: m.distance)
                # only the top 10 matches
                matches_to_show = matches_to_show[:10]
                match_quality = len(matches_to_show) / len(these_title_matches) * 100
                matches_img = cv2.drawMatches(
                    img1=cam_title_area,
                    keypoints1=kp_cam_title,
                    img2=ref_title,
                    keypoints2=kp_ref_title,
                    matches1to2=matches_to_show,
                    outImg=None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                # draw match_quality text on matches_img
                cv2.putText(matches_img, f'{match_quality:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('matches', matches_img)
                cv2.waitKey(0)        

        ratio = 0.7
        good_matches_full_card  = [m for m, n in matches_full_card if m.distance < ratio * n.distance]
        good_matches_for_titles = [[m for m, n in matches if m.distance < ratio * n.distance] for matches in matches_titles]
        
        # # After obtaining the good matches, calculate the homography
        # src_pts_full_card = np.float32([kp_cam[m.queryIdx].pt for m in good_matches_full_card]).reshape(-1, 1, 2)
        # dst_pts_full_card = np.float32([kp_tarot[m.trainIdx].pt for m in good_matches_full_card]).reshape(-1, 1, 2)

        # # Find homography using RANSAC
        # M, mask = cv2.findHomography(srcPoints=src_pts_full_card, dstPoints=dst_pts_full_card, method=cv2.RANSAC, ransacReprojThreshold=5.0)

        # # Filter out the outliers using the mask
        # good_matches_full_card = np.array(good_matches_full_card)[mask.ravel() == 1].tolist()

        match_quality_full_card = 0 if len(matches_full_card) == 0 else len(good_matches_full_card) / len(matches_full_card) * 100
        # Max quality for everything in good_matches_for_titles by the above metric
        match_quality_title_area = max([0 if len(matches) == 0 else len(good_matches) / len(matches) * 100 for matches, good_matches in zip(matches_titles, good_matches_for_titles)])

        match_quality = match_quality_full_card #(match_quality_full_card + match_quality_title_area) / 2

        if params['debug_show']:
            # Sort the matches by distance (lower is better)
            matches = sorted(good_matches_full_card, key=lambda x: x.distance)

            matches_img = cv2.drawMatches(
                img1=cam_image,
                keypoints1=kp_cam,
                img2=cards_keypoints[tarot_card]['image'],
                keypoints2=kp_tarot,
                matches1to2=matches[:20], 
                outImg=None,
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
            )
            # draw the text "Q: <match_quality>" on the top left corner of the matches_img
            cv2.putText(matches_img, f'Q: {match_quality:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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

def main():
    # vid = cv2.VideoCapture(0) # built in
    # vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # vid.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    # vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # vid.set(cv2.CAP_PROP_FPS, FPS)
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # stream = LiftedStream(video_capture_generator(vid))
    # stream = LiftedStream(oak_capture_generator())
    # stream = ThreadedCamera(0, cv2.CAP_DSHOW)
    # stream.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    # stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    vid = ThreadedCamera(0, cv2.CAP_DSHOW)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    output = {}
    process_stream = Pipeline() \
        .map(pipeline.otsu_threshold(params)) \
        .map(pipeline.contours(output)) \

    should_continue = True

    try:
        while should_continue:
            time.sleep(1 / FPS)

            raw_frame = vid.latest()
            frame = process_stream(raw_frame)

            contours = output['contours']
            # sort by center x
            contours = sorted(contours, key=lambda contour: np.mean(contour[:, 0, 0]))

            frame_to_show = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if params['show_threshold'] else raw_frame
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
            highDefFrame = raw_frame

            # for each approx
            for approx in approximations:
                # show image

                # highDefImage = four_point_transform(highDefFrame, approx.reshape((4, 2)).astype(np.float32))
                highDefImage = perspective_transform(highDefFrame, approx)
                cv2.imshow('highDefImage', highDefImage)
                # find the best match
                t0 = time.time()
                result = find_best_text_match(highDefImage)
                t1 = time.time()
                ms = (t1 - t0) * 1000
                print(f'find_best_text_match took {ms} milliseconds')

                matches.append(result[0])
                # t0 = time.time()
                # st_match = find_best_sentence_transformer_match(highDefImage)
                # t1 = time.time()
                # ms = (t1 - t0) * 1000
                # print(f'find_best_sentence_transformer_match took {ms} milliseconds')

                # # intersection of result & st_match
                # intersection = [match for match in result if match in st_match]

                # if len(intersection) > 0:
                #     matches.append(intersection[0])
            # for each bounding box, extract high def image and run detector.detectAndCompute
            # for (x, y, w, h) in boundingBoxes:
            #     # extract high def image
            #     highDefImage = highDefFrame[y:y+h, x:x+w]

            #     result = find_best_match(highDefImage)
            #     matches.append(result)

                # # run detector.detectAndCompute
                # kp, des = detector.detectAndCompute(image=highDefImage, mask=None)

                # # cards_keypoints is a dictionary like:
                # # {
                # #     'magician': {'keypoints': [], 'descriptors': [], 'image': <image>},
                # #     'high_priestess': {'keypoints': [], 'descriptors': [], 'image': <image>},
                # #     ...
                # # }
                # # use matcher (flann) to find the most similar tarot card
                # for card_name, card_data in cards_keypoints.items():
                #     matches = matcher.knnMatch(des, card_data['descriptors'], k=2)
                #     good = []
                #     # for m, n in matches:
                #     #     if m.distance < 0.7 * n.distance:
                #     #         good.append([m])
                #     # # draw the matches
                #     # print(card_name, len(good))
                #     matchesMask = [[0,0] for i in range(len(matches))]
                #     # ratio test as per Lowe's paper
                #     for i,(m,n) in enumerate(matches):
                #         if m.distance < 0.7*n.distance:
                #             matchesMask[i]=[1,0]
                #     draw_params = dict(matchColor = (0,255,0),
                #                        singlePointColor = (255,0,0),
                #                        matchesMask = matchesMask,
                #                        flags = cv2.DrawMatchesFlags_DEFAULT)
                #     img3 = cv2.drawMatchesKnn(highDefImage,kp,card_data['image'],card_data['keypoints'],matches,None,**draw_params)
                #     cv2.imshow('image 3', img3)
            cv2.drawContours(image=frame_to_show, contours=approximations, contourIdx=-1, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            # for each approximation i, draw the text matches[i] just below bounding box
            for i, (x, y, w, h) in enumerate(boundingBoxes):
                cv2.putText(frame_to_show, matches[i], (x + 50, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
                              
            
            cv2.imshow("Found Cards", cv2.resize(frame_to_show, (720, 405)))

            print(matches)

    except Exception as e:
        print("Error:", e)
        print(traceback.format_exc())

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

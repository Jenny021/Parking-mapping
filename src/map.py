import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from slot import Slot
from collections import Counter


slots = {
    1: [Slot((869, 469), "Turn left, slot on right"), Slot((873, 655), "Turn right, slot on left")],
    2: [Slot((872,377), "Turn left, slot on right"), Slot((876,743), "Turn right, slot on left")],
    3: [Slot((871,282), "Turn left, slot on right"), Slot((876,837), "Turn right, slot on left"), Slot((1099,360), "Go straight, slot on left"), Slot((1155,761), "Go straight, slot on right")],
    4: [Slot((877,926),  "Turn left, slot on left"), Slot((466,931), "Turn right, slot on right"),  Slot((1198,363), "Go straight, slot on left"), Slot((1244,759), "Go straight, slot on right")],
    5: [Slot((878,1017), "Turn left, slot on left"), Slot((466,1023), "Turn right, slot on right"), Slot((1311,372), "Go straight, slot on left"), Slot((1337,759), "Go straight, slot on right")],
    6: [Slot((878,1107), "Turn left, slot on left"), Slot((469,1111), "Turn right, slot on right"), Slot((1412,339), "Go straight, slot on left"), Slot((1428,759), "Go straight, slot on right")],
    7: [Slot((881,1199), "Turn left, slot on left"), Slot((469,1204), "Turn right, slot on right"), Slot((1510,339), "Go straight, slot on left"), Slot((1519,759), "Go straight, slot on right")],
    8: [Slot((881,1294), "Turn left, slot on left"), Slot((468,1293), "Turn right, slot on right"), Slot((1600,339), "Go straight, slot on left"), Slot((1608,759), "Go straight, slot on right")],
    9: [Slot((881,1379), "Turn left, slot on left"), Slot((465,1385), "Turn right, slot on right"), Slot((1715,339), "Go straight, slot on left"), Slot((1715,761), "Go straight, slot on right")],
    10: [Slot((468,1481), "Turn right, slot on right"), Slot((1809,333), "Go straight, slot on left"), Slot((1837,759), "Go straight, slot on right")],
    11: [Slot((1940,338), "Turn left"), Slot((1926,757), "Turn left")],
    12: [Slot((2033,341), "Go straight, slot on left"), Slot((2021,759), "Go straight, slot on right")],
    13: [Slot((2126,338), "Go straight, slot on left"), Slot((2129,763), "Go straight, slot on right")],
    14: [Slot((2216,338), "Go straight, slot on left"), Slot((2215,765), "Go straight, slot on right")],
    15: [Slot((2375,341), "Go straight, slot on left"), Slot((2311,768), "Go straight, slot on right")],
    16: [Slot((2401,764), "Go straight, slot on right"), Slot((2490,341), "Go straight, slot on leftt")],
    17: [Slot((2575,343), "Go straight, slot on left")],
    22: [Slot((2548,1308), "Go straight, turn right, turn right, slot on left")],
    23: [Slot((2462,1307), "Go straight, turn right, turn right, slot on left")],
    24: [Slot((2387,968), "Go straight, turn right, turn right, slot on right"), Slot((2370,1305), "Go straight, turn right, turn right, slot on left")],
    25: [Slot((2300,971), "Go straight, turn right, turn right, slot on right"), Slot((2281,1307), "Go straight, turn right, turn right, slot on left")],
    26: [Slot((2206,966), "Go straight, turn right, turn right, slot on right"), Slot((2186,1303), "Go straight, turn right, turn right, slot on left")],
    27: [Slot((2117,969), "Go straight, turn right, turn right, slot on right"), Slot((2097,1305), "Go straight, turn right, turn right, slot on left")],
    28: [Slot((2026,969), "Go straight, turn right, turn right, slot on right"), Slot((2008,1305), "Go straight, turn right, turn right, slot on left")],
    29: [Slot((1927,960), "Go straight, turn right, turn right, slot on right"), Slot((1910,1305), "Go straight, turn right, turn right, slot on left")],
    30: [Slot((1828,962), "Go straight, turn right, turn right, slot on right"), Slot((1822,1305), "Go straight, turn right, turn right, slot on left")],
    31: [Slot((1708,962), "Go straight, turn right, turn right, slot on right"), Slot((1732,1305), "Go straight, turn right, turn right, slot on left")],
    32: [Slot((1617,964), "Go straight, turn right, turn right, slot on right"), Slot((1640,1305), "Go straight, turn right, turn right, slot on left")],
    33: [Slot((1528,964), "Go straight, turn right, turn right, slot on right"), Slot((1552,1307), "Go straight, turn right, turn right, slot on left")],
    34: [Slot((1328,964), "Go straight, turn right, turn right, slot on right"), Slot((1461,1305), "Go straight, turn right, turn right, slot on left")],
    35: [Slot((1437,964), "Go straight, turn right, turn right, slot on right"), Slot((1368,1307), "Go straight, turn right, turn right, slot on left")],
    36: [Slot((1346,964), "Go straight, turn right, turn right, slot on right"), Slot((1274,1307), "Go straight, turn right, turn right, slot on left")],
    37: [Slot((1190,1310), "Go straight turn right, turn right last to the left")],
}

def most_common(arr):
    string_counts = Counter(arr)
    most_common = string_counts.most_common(1)
    return most_common[0][0] if most_common else None

def video_loader(path):

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    output_directory = 'output_frames'
    os.makedirs(output_directory, exist_ok=True)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    nb_frames = 5;

    selected_frames = random.sample(range(0, total_frames), nb_frames)

    frames_names = []
    for frame_number in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            filename = f"frame_{frame_number}.jpg"
            frame_path = os.path.join(output_directory, filename)
            cv2.imwrite(frame_path, frame)
            frames_names.append(frame_path)

    sentences = []
    for idx, frame in enumerate(frames_names):
        sentences.append(map_analyser(frame))
    answer = most_common(sentences)
    if (answer == None):
        print("No slot available")
    else:
        print(answer)
    cap.release()
    cv2.destroyAllWindows()


def map_analyser(filename):
    img1 = cv2.imread("./map_without_cars.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    n = 100
    if len(good)>n:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w = img1.shape
        M_inverse = np.linalg.inv(M)
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), n) )

    img5 = cv2.warpPerspective(img2, M_inverse, (img1.shape[1], img1.shape[0]))
    empty_slots = check_empty(img1, img5)
    if len(empty_slots) == 0:
        print("No available slots")
        return;
    for slot in empty_slots:
        br_corner = slot.coord
        if br_corner[0] < 900:
            br_corner = (br_corner[0] + 182, br_corner[1] + 82)
        else:
            br_corner = (br_corner[0] + 82, br_corner[1] + 182)
        cv2.rectangle(img5, slot.coord, br_corner, 255, thickness=5)
    desired_width = 1920
    desired_height = 1080
    resized_image = cv2.resize(img5, (desired_width, desired_height))
    cv2.imshow('img', resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return empty_slots[0].sentence;
    return empty_slots[0].sentence;

def check_horizontal(slot, emap, status):
    x = slot[0] + 10
    y = slot[1]
    log = 0
    for i in range(4):
        y = slot[1]
        for j in range(3):
            pxl = emap[y, x]
            pxl2 = status[y, x]
            if (pxl2 > pxl + 5 or pxl2 < pxl - 5):
                log+= 1
            y += 23
        x += 40
    if log >= 8:
        return 0
    return 1

def check_vertical(slot, emap, status):
    x = slot[0]
    y = slot[1]
    log = 0
    for i in range(4):
        for j in range(3):
            pxl = emap[y, x]
            pxl2 = status[y, x]
            if (pxl2 > pxl + 5 or pxl2 < pxl - 5):
                log += 1
            x += 23
        y += 40
        x = slot[0]
    if log >= 8:
        return 0
    return 1

def check_empty(emap, status):
    L = [];
    for i in slots:
        for elem in slots[i]:
            res = 0
            if elem.coord[0] < 900:
                res = check_horizontal(elem.coord, emap, status)
            else:
                res = check_vertical(elem.coord, emap, status)
            if res == 1:
                L.append(elem)
    return L
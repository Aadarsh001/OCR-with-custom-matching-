"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np
from IPython import embed
import json
import pickle
import sys

sys.setrecursionlimit(100000)
visited = []
coord = []
image = []
image1 = []
max_int = float('inf')
def get_threshold(img):
    bins = 10
    bin_values = []
    rg = 255/bins
    mvalue = rg
    for i in range(bins): 
        bin_values.append(int(mvalue))
        mvalue += rg
    histograms = [1]*bins
    for i in range(img.shape[0]): #histogram
        for j in range(img.shape[1]):
            for k in range(len(bin_values)):
                if img[i][j]<bin_values[k]:
                    histograms[k]+=1
                    break
    peaks = []
    
    for i in range(1,len(histograms)):
        if histograms[i]>histograms[i-1]:
            if i == len(histograms)-1:
                peaks.append(bin_values[i])
            elif histograms[i]>histograms[i+1]:
                peaks.append(bin_values[i])
        
    if len(peaks)>1:
        threshold = (peaks[-1] + peaks[-2])/2
    else:
        threshold = 178

    # print("threshold:{}".format(threshold))

    return threshold
def gaussian(img,g_size):
    shape = (img.shape[0]-(g_size-1),img.shape[1]-(g_size-1))
    gaussian = np.zeros(shape)
    kernel = np.array([[1]*g_size]*g_size)
    for i in range(shape[0]):
        for j in range(shape[1]):
            gaussian[i][j] = (np.sum(kernel * img[i:i+g_size, j:j+g_size]))/(g_size*g_size)


    
    return gaussian



def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def conv_to_binary(img):
    thres = get_threshold(img)
    # thres = 178

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            if img[i][j] < thres:
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    features = enrollment(characters)

    detection(test_img)
    
    return recognition(test_img, features)


import copy
def get_features(img):
   
    img1 = copy.deepcopy(img)
    img1_padded = np.zeros((img1.shape[0]+10,img1.shape[1]+10))
    pcolor = 255
    g_size = 5
    img1_padded[0:g_size] = pcolor
    img1_padded[-g_size:] = pcolor
    img1_padded[:,0:g_size] = pcolor
    img1_padded[:,-g_size:] = pcolor
    img1_padded[g_size:-g_size,g_size:-g_size] = img1
    img1 = img1_padded
    img1 = gaussian(img1,g_size)
    shape = img1.shape

    while shape[0]<20 or shape[1]<20:
        shape = (shape[0]*2,shape[1]*2)
        
        img1 = cv2.resize(img1, shape, interpolation = cv2.INTER_AREA)

   
    img1 = img1_padded.astype('uint8')
    ext = cv2.SIFT_create()
    
    keypoint, descriptor = ext.detectAndCompute(img1, None)
    try:
        if not descriptor:
            descriptor = []
    except:
        pass
    
    return descriptor
def enrollment(characters):
    features = {}
    filt = [1,-1]
    for char in characters:
        name,char = char
        char = conv_to_binary(char)
        dims = get_loc(char)[0]
        char = char[dims[0]:dims[2],dims[1]:dims[3]]
        
        features[name] = get_features(char)
    return features
    


def detection(img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.

    return get_loc(img)
    

    


def get_loc(img):
    
    global visited
    global coord
    global image1
    image1 = img
    shape = image1.shape
    visited = np.zeros(shape)
    i=0
    j=0
    cnt = 0
    characters = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if image1[i][j]<90 and not visited[i][j]:
                dfs(i, j, shape)
                char = coord
                
                x_char = [item[0] for item in char]
                y_char = [item[1] for item in char]
                xmin, ymin, xmax, ymax = min(x_char), min(y_char), max(x_char), max(y_char)
                if xmax-xmin<=2 or ymax-ymin<=2: #To avoid Noise
                    continue
                characters.append([xmin,ymin,xmax,ymax])
                # cv2.imwrite("detected/char_{}.jpg".format(cnt), image1[xmin:xmax,ymin:ymax])
                
                coord = []
                cnt += 1
    return characters
            


def dfs(i , j, shape):
    global visited
    global coord
    if image1[i][j]>165 or visited[i][j]:
        return []
    visited[i][j] = True
    if i<shape[0]-1:
        dfs(i+1, j, shape)
    if i>0:
        dfs(i-1, j, shape)
    if j<shape[1]-1:
        dfs(i, j+1, shape)
    if j>0:
        dfs(i, j-1, shape)

    coord.append([i,j])
    
def euclidean(d1,d2):
    dist = np.subtract(d1,d2)
    dist_sum_sq = np.dot(dist.T,dist)

    
    return np.sqrt(dist_sum_sq)

def calc_dist(f1,f2):
    min_dist = max_int
    total_dist = 0
    distances = {}
    for i in range(len(f1)):
        distances[i] = []
        for j in range(len(f2)):
            distances[i].append(euclidean(f1[i],f2[j]))
    match = 0
    min_dist = max_int
    fc = 0
    tf = len(distances)
    for kf in distances:
        distances[kf].sort()
        if len(distances[kf])>1 and distances[kf][0]/distances[kf][1]<0.8:#and distances[kf][0]/distances[kf][1]>0.3
            match += 1
            total_dist += distances[kf][0]
    
    if total_dist:
        return match,total_dist/match
    else:
        return match,max_int



def recognition(test_img, features):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    threshold = 200
    bbox_coord = get_loc(test_img)
    results = []
    count = 0
    for bbox in bbox_coord:
        # print("For char_{}".format(count))
        count += 1
        char = test_img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        ft_test = get_features(char)
        dist = threshold
        recog = "UNKNOWN"
        for name in features:
            fte = features[name]

            if not len(fte) or not len(ft_test):
                continue
            
            nmatch,tdist = calc_dist(fte,ft_test)
            # print(nmatch,tdist,name)
            if tdist < dist and nmatch>3:
                recog = name
                dist = tdist
        
        # print(dist,recog)
        tres = {}
        tres['bbox'] = [bbox[1],bbox[0],bbox[3]-bbox[1],bbox[2]-bbox[0]]
        tres['name'] = recog
        results.append(tres)
        
        # print("---------------------------------------------------------------------")

    save_results(results,"./data")
    return results


    


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = []
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    all_character_imgs_2 = glob.glob("./characters/*")
    try:
        for each_character in all_character_imgs :
            character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
            characters.append([character_name, read_image(each_character, show=False)])
    except:
        for each_character in all_character_imgs_2 :
            character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
            characters.append([character_name, read_image(each_character, show=False)])
    global image
    try:
        image = read_image(args.test_img)
    except:
        image = read_image("test_img.jpg")
    image = conv_to_binary(image)
    results = ocr(image, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()

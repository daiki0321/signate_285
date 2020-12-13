import os
import sys
import cv2
import numpy as np

color2index = {
    (0,	    0,	    255) : (255,	    0,	    0),
    (193,	214,	 0)  : (0,	214,	 193),
    (180,	0,	    129) : (129,	0,	    180),
    (255,	121,	166) : (166,	121,	255),
    (255,	0,	    0)   : (0,	0,	    255),
    (65,	166,	1)   : (1,	166,	65),
    (208,	149,	1)   : (1,	149,	208),
    (255,	255,	0)   : (0,	255,	255),
    (255,	134,	0)   : (0,	134,	255),
    (0,	    152,	225) : (225,	    152,	0),
    (0,	    203,	151) : (151,	    203,	0),
    (85,	255,	50)  : (50,	255,	85),
    (92,	136,	125) : (125,	136,	92),
    (69,	47,	    142) : (142,	47,	    69),
    (136,	45,	    66)  : (66,	45,	    136),
    (0,  	255,	255) : (255,  	255,	0),
    (215,	0,	    255) : (255,	0,	    215),
    (180,	131,	135) : (135,	131,	180),
    (81,	99,	    0)   : (0,	99,	    81),
    (86,	62,	    67)  : (67,	62,	    86),
}

def im2index(im):
    """
    turn a 3 channel RGB image to 1 channel index image
    """
    assert len(im.shape) == 3
    height, width, ch = im.shape
    #height, width, ch = (1216, 1936, 3)
    assert ch == 3
    #m_lable = np.zeros((height, width, 3), dtype=np.uint8)
    m_lable = np.full((height, width, 3), 255,dtype=np.uint8)
    for w in range(width):
        for h in range(height):
            #print(im)
            b, g, r = im[h, w, :]
            m_lable[h, w, :] = color2index[(r, g, b)]
    return m_lable

train_folder = sys.argv[1]
train_images = os.listdir(train_folder)

for i in range (0, len(train_images)):

    base, ext = os.path.splitext(train_images[i])
    if (ext != ".png"): 
        continue

    print(train_images[i])
    im = cv2.imread(os.path.join(train_folder, train_images[i]), 1)

    #print(im)

    #im.shape = (1216, 1936, 3)

    print(im.shape)

    label = im2index(im)

    os.makedirs("./output_train_black", exist_ok=True)

    cv2.imwrite("./output_train_black/"+train_images[i] ,label)


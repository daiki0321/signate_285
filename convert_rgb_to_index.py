import os
import sys
import cv2
import numpy as np

color2index = {
    (0,	    0,	    255) : 0,
    (193,	214,	 0)  : 1,
    (180,	0,	    129) : 2,
    (255,	121,	166) : 3,
    (255,	0,	    0)   : 4,
    (65,	166,	1)   : 5,
    (208,	149,	1)   : 6,
    (255,	255,	0)   : 7,
    (255,	134,	0)   : 8,
    (0,	    152,	225) : 9,
    (0,	    203,	151) : 10,
    (85,	255,	50)  : 11,
    (92,	136,	125) : 12,
    (69,	47,	    142) : 13,
    (136,	45,	    66)  : 14,
    (0,  	255,	255) : 15,
    (215,	0,	    255) : 16,
    (180,	131,	135) : 17,
    (81,	99,	    0)   : 18,
    (86,	62,	    67)  : 19,
    (255,	255,   255)  : 255,
}

def im2index(im):
    """
    turn a 3 channel RGB image to 1 channel index image
    """
    assert len(im.shape) == 3
    height, width, ch = im.shape
    #height, width, ch = (1216, 1936, 3)
    assert ch == 3
    m_lable = np.zeros((height, width, 1), dtype=np.uint8)
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

    os.makedirs("./output_deeplab_seg", exist_ok=True)

    cv2.imwrite("./output_deeplab_seg/"+train_images[i] ,label)


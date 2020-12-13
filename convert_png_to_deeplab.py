import os
import json
from collections import OrderedDict
import pprint

import cv2
import numpy as np

TRAIN_ANNOTATE_DIR="./seg_train_annotations/"
OUTPUT_DIR="./output_deeplab/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open('submit_correct.json') as f:
    df = json.load(f)

#pprint.pprint(df, width=40)

for df_key in df:
    image_name = df_key

    print(image_name)

    im = cv2.imread(TRAIN_ANNOTATE_DIR+os.path.splitext(image_name)[0]+'.png', 1)

    assert len(im.shape) == 3
    height, width, ch = im.shape
    #height, width, ch = (1216, 1936, 3)
    assert ch == 3

    print(width)
    print(height)

    for w in range(0,width-1):
        for h in range(0, height-1):
            #print(im)

            if (np.array_equal(im[h, w, :], im[h+1, w, :]) and np.array_equal(im[h, w, :], im[h, w+1, :])):
                im[h, w, :] = im[h, w, :]
            else:
                im[h, w, :] = (255,255,255)
            
            #if w == 0:
            #    print(im[h, w, :])

    cv2.imwrite(OUTPUT_DIR+os.path.splitext(image_name)[0]+'.png' ,im)




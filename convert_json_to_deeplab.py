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

    for label_name_key in df[image_name]:
        label_name = label_name_key

        y_prev = 0

        for y_key in df[image_name][label_name]:
            y = y_key

            for x_key in df[image_name][label_name][y]:

                x0, x1 = x_key

                if y_prev != int(y)-1:
                    im[int(y), x0:x1, :] = (255,255,255)
                else:
                    im[int(y), x0, :] = (255,255,255)
                    im[int(y), x1, :] = (255,255,255)

            y_prev = int(y)
    
    cv2.imwrite(OUTPUT_DIR+os.path.splitext(image_name)[0]+'.png' ,im)




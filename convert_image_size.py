import os
import sys
import cv2

folder_name = sys.argv[1]

print(folder_name)

os.makedirs(folder_name+"_resize_500x500", exist_ok=True)

images = os.listdir(folder_name)

for i in range (0, len(images)):

    im = cv2.imread(os.path.join(folder_name, images[i]))

    height, width, _ = im.shape

    half_image = cv2.resize(im,(500,500), interpolation = cv2.INTER_NEAREST)

    cv2.imwrite(folder_name+"_resize_500x500"+"/"+images[i] ,half_image)


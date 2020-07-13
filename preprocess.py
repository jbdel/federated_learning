import cv2, glob, os
import numpy as np
import sys
import pickle
from tqdm import tqdm

def scaleRadius(img,scale):
    x=img[img.shape[0]//2,:,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)

scale = 256
files = glob.glob("/media/jb/DATA/retina/train/*.jpeg")
for i, f in enumerate(tqdm(files)):
    try:
        outfile = "out_train/" + os.path.basename(f).split(".")[0]
        if os.path.exists(outfile+".npy"):#and os.path.getsize(outfile+".npy") == 393344:
            continue
        a = cv2.imread(f)
        a = scaleRadius(a, scale)
        b = np.zeros(a.shape)
        cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
        aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128) * b + 128 * (1 - b)
        aa = cv2.resize(aa, (256, 256))
        # cv2.imwrite(str(scale) + "_" + f, aa)
        np.save(outfile, np.array(aa).astype(np.uint8))
    except cv2.error:
        print("Unexpected error:", sys.exc_info()[0])
        print(f)

# test 2 fail : /media/jb/DATA/retina/test/25313_right.jpeg // /media/jb/DATA/retina/test/27096_right.jpeg
# train 1 fail : /media/jb/DATA/retina/train/492_right.jpeg

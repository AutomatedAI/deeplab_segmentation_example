
from prediction_support import *
import cv2
cityscapes_viz = True
model_dir = 'data/coco.tar.gz'

# Set up GPU, and model  
device='/gpu:0'
gpu_frac = .09 
# init Model in GPU
MODEL = DeepLabModel(model_dir,device,gpu_frac)

# load a testing Image (RGB image)
orignal_im = Image.open('data/bike.jpg' )

# run the model on the image
img = MODEL.run(orignal_im)

while (cv2.waitKey(10) != ord('q')):
    cv2.imshow("output img",img)

cv2.destroyAllWindows()

resized_im, seg_map = MODEL.run_mask(orignal_im)
# Visulize the image as a test
vis_segmentation(resized_im, seg_map)
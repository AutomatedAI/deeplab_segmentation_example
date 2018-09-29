
from prediction_support import *
import cv2
cityscapes_viz = True
model_dir = 'data/deeplab_model_coco.tar.gz'

# Set up GPU, and model  
device='/gpu:0'
gpu_frac = .09 
# init Model in GPU
MODEL = DeepLabModel(model_dir,device,gpu_frac)

# load a testing Image (RGB image)
orignal_im = Image.open('data/bike.jpg' )

# run the model on the image
resized_im, seg_map = MODEL.run(orignal_im)

# make a segmented image with the result
img = make_sending_seg(resized_im, seg_map)

while (cv2.waitKey(10) != ord('q')):
    cv2.imshow("output img",img)

cv2.destroyAllWindows()

# Visulize the image as a test
vis_segmentation(resized_im, seg_map)
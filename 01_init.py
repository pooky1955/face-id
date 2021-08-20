import os
from mtcnn_cv2 import MTCNN
from keras_vggface.vggface import VGGFace
import cv2
model = VGGFace(model="resnet50",inc lude_top=False)
detector = MTCNN()

print("VGGFace model loaded")

print("MTCNN detector loaded")

if not os.path.exists('faces'):
  os.mkdir('faces')

print("Succesfully init")



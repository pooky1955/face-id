import os
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
import cv2
model = VGGFace(model="resnet50",include_top=False)

print("VGGFace model loaded")

detector = MTCNN()
print("MTCNN detector loaded")

if not os.path.exists('faces'):
  os.mkdir('faces')

print("Succesfully init")



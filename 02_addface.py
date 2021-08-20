import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import pickle
from util import get_snapshot, extract_face, show_face, get_embedding, save_embedding, extractor, detector

GREEN = (0,255,0)
''' Main Code '''
def add_face():
  name = input("Enter a name for the face ")

  # checks if already exists
  if os.path.exists(f"faces/{name}.pickle"):
    print("Embedding already exists")
    if input("Overwrite the file? (y,[n])") != 'y':
      print("Skipping")
      return 


  # get snapshot and coordinates of face
  snapshot, coords = get_snapshot()
  # get face region reshaped
  face = extract_face(snapshot,coords)
  print("Got face snapshot for ",name)

  show_face(face)

  # get embedding
  embedding = get_embedding(face)
  print("Got embedding")


  #save embedding
  save_embedding(embedding,name)


if __name__ == "__main__":
    
  while True:
    add_face()
    if input("Add another face? (y,[n])") != 'y':
      break


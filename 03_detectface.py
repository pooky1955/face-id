import os
import pickle
from scipy.spatial.distance import cosine
import cv2
from util import extract_face, get_embedding, detector, extractor

cap = cv2.VideoCapture(0)
GREEN = (0,255,0)

def load_batch(filepaths,folder_prefix=None):
  objs = []
  if folder_prefix != None:
    filepaths = ['/'.join([folder_prefix,filepath]) for filepath in filepaths]
  for filepath in filepaths:
    with open(filepath,"rb") as f:
      obj = pickle.load(f)
      objs.append(obj)

  return objs

def compare(candidate_embed,reference_embeds,labels,threshold=1):
  distances = [(label, cosine(candidate_embed,reference_embed)) for label,reference_embed in zip(labels,reference_embeds)]
  distances = filter(lambda x: x[1] < threshold,distances)  
  distances = sorted(distances,key=lambda x: x[1])
  show_distances(distances)
  return distances[0]

def show_distances(distances):
  text = ' | '.join([f"{distance[0]} - {distance[1]:.2f}" for distance in distances])
  print(text)

embed_files = os.listdir("faces")
reference_embeds = load_batch(embed_files,folder_prefix="faces")
labels = [embed_file.split('.')[0] for embed_file in embed_files]


while True:
  is_ok, frame = cap.read()
  rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  faces = detector.detect_faces(rgb_frame)
  print(len(faces))
  if len(faces) == 1:
    x,y,w,h = faces[0]["box"]
    cv2.rectangle(frame,(x,y),(x+w,y+h),GREEN,4)

    face_region = extract_face(rgb_frame,(x,y,w,h))
    candidate_embed = get_embedding(face_region)

    label, distance = compare(candidate_embed,reference_embeds,labels)
    if distance > 0.5:
      label = "unknown"
    cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,GREEN,1,cv2.LINE_AA)
  
  
  if 0xff & cv2.waitKey(1) == ord('q'):
    break



  cv2.imshow("Face ID",frame)



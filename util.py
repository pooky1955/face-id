from mtcnn_cv2 import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import cv2
import matplotlib.pyplot as plt
import pickle
detector = MTCNN()
extractor = VGGFace(model="resnet50",include_top=False)

GREEN = (0,255,0)

def get_snapshot():
  ''' uses cv2 video capture to return snapshot and coords '''
  cap = cv2.VideoCapture(0)
  while True:
    is_ok, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    faces = detector.detect_faces(rgb_frame)
    if len(faces) == 1:
      x,y,w,h = faces[0]["box"]
      cv2.rectangle(frame,(x,y),(x+w,y+h),GREEN,4)

      # key press to take a picture
      if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.destroyAllWindows()
            return rgb_frame, (x,y,w,h)
    
    cv2.imshow("Press P to take a picture",frame)



def extract_face(snapshot,coords,required_size=(224,224)):
  ''' extracts face given frame and coords '''
  # select face region boundary
  x,y,w,h = coords
  face_region = snapshot[y:y+h,x:x+w]

  # resize image
  reshaped = cv2.resize(face_region,required_size)

  return reshaped

def show_face(face):
  ''' shows face in plt'''
  plt.title("A beautiful face")
  plt.imshow(face)
  plt.show()

def get_embedding(face):
  ''' face to embedding '''
  faces = face.reshape((1,*face.shape)).astype("float32")
  preprocessed_faces = preprocess_input(faces)
  embedding = extractor.predict(preprocessed_faces)[0]

  return embedding



def save_embedding(embedding,name):
  embed_path = f"faces/{name}.pickle"
  with open(embed_path,"wb") as f:
    pickle.dump(embedding,f)
  print("Dumped embedding")

def load_batch(filepaths,folder_prefix=None):
  ''' loads multiple pickle files '''
  objs = []
  if folder_prefix != None:
    filepaths = ['/'.join([folder_prefix,filepath]) for filepath in filepaths]
  for filepath in filepaths:
    with open(filepath,"rb") as f:
      obj = pickle.load(f)
      objs.append(obj)

  return objs



import os
import cv2
import numpy as np
from PIL import Image

def read_image(filename):
    img = cv2.imread(filename)
    return img

def select_random_pair(sample_size,data_size,pair_size = 2):
    alist = np.random.choice(np.arange(data_size),size=(sample_size,pair_size))
    return alist

def get_data(data_path, data_path_imposter,sample_size):
    X_similar_pair = []
    y_similar = []
    X_dissimilar_pair = []
    y_dissimilar = []
    for sub_file in os.listdir(data_path):
      print(sub_file)
      print('----Making similar pair----')
      filelist = os.listdir(os.path.join(data_path,sub_file))
      random_list = select_random_pair(sample_size,len(filelist))
      for i in random_list:
          ind1 = i[0]
          ind2 = i[1]
          if ind1 == ind2:
              sample_size_1 = sample_size-1
              continue
          img1 = read_image(os.path.join(data_path,sub_file,filelist[ind1]))
          img2 = read_image(os.path.join(data_path,sub_file,filelist[ind2]))
          X_similar_pair.append([img1,img2])
          y_similar.append([1])
      print('Number of similar pair: ', sample_size_1)
      print('----Making dissimilar pair----')
      for i in sample_size:
          filelist_1 = os.listdir(os.path.join(data_path,sub_file))
          filelist_2 = os.listdir(os.path.join(data_path_imposter,sub_file))
          ind1 = np.random.randint(len(filelist_1))
          ind2 = np.random.randint(len(filelist_2))
          img1 = read_image(os.path.join(data_path,sub_file,filelist_1[ind1]))
          img2 = read_image(os.path.join(data_path_imposter,sub_file,filelist_2[ind2]))
          X_dissimilar_pair.append([img1,img2])
          y_dissimilar.append([0])
      print('Number of dissimilar pair: ', sample_size)
      print('---Done---')
      print()
    X = np.concatenate([X_similar_pair, X_dissimilar_pair], axis=0)
    Y = np.concatenate([y_similar, y_dissimilar], axis=0)
    print('Total of pair sets: ', len(X))
    return X, Y

def crop_faces_helper(img,face_bbox,margin):
    x1, y1, width, height = face_bbox

    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    x1 = x1 - margin / 2 if x1 - margin / 2 > 0 else 0
    y1 = y1 - margin / 2 if y1 - margin / 2 > 0 else 0
    x2 = x2 + margin / 2 if x2 + margin / 2 < img.shape[1] else img.shape[1]
    y2 = y2 + margin / 2 if y2 + margin / 2 < img.shape[0] else img.shape[0]
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

    # extract the face
    face = img[y1:y2, x1:x2]
    # resize pixels to the model size
    face = Image.fromarray(face)
    return face

def extract_faces(img_array, detector,detect_multiple_faces,threshold, image_size=160, margin=44):
    faces_list, bbox = [], []
    # convert channel

    results = detector.detect_faces(img_array)

    for face in results:
        confidence = face['confidence']
        if confidence < threshold:
            continue
        face_bbox = face['box']
        face = crop_faces_helper(img_array,face_bbox,margin)
        face = face.resize((image_size, image_size))
        face = np.asarray(face)
        faces_list.append(face)
        bbox.append(face_bbox)
        if not detect_multiple_faces:
            return faces_list, bbox

    return faces_list, bbox

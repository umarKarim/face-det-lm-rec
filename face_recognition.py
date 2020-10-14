import tensorflow as tf 
from options import Options 
import os
import PIL.Image as Image 
import numpy as np 
import cv2 
from face_detection import FaceDetector 


class FaceRecognizer():
    def __init__(self):
        Opts = Options() 
        opts = Opts.opts 
        self.recog_th = opts.recog_th 
        self.face_database = opts.face_database
        self.face_model_path = opts.face_recog_model_path 
        self.feature_extractor = tf.lite.Interpreter(self.face_model_path)
        self.feature_extractor.allocate_tensors() 
        self.input_details = self.feature_extractor.get_input_details() 
        self.output_details = self.feature_extractor.get_output_details() 

        self.in_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']

        self.FaceDetector = FaceDetector()
        self.known_names, self.known_embeddings = self.get_embeddings_dir()

    def get_embeddings_dir(self):
        file_names = os.listdir(self.face_database)
        im_names = [x for x in file_names if x.endswith('.png') or x.endswith('.jpg')
                    or x.endswith('.JPG') or x.endswith('.jpeg')] 
        known_names = sorted([self.face_database + x for x in im_names])
        known_features = [] 
        for im_name in known_names:
            im_emb, face_crop = self.get_embedding_1(im_name, is_array=False)
            known_features += [im_emb]
        return known_names, known_features 

    def get_embedding_1(self, im_name, is_array=True):
        if is_array is False:
            crop_im = self.load_image(im_name)
        else:
            crop_im = im_name 
        crop_im = cv2.resize(crop_im, (self.in_shape[1], self.in_shape[2]), interpolation=cv2.INTER_LINEAR)
        crop_im = cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB)
        crop_im = np.expand_dims(crop_im, axis=0)
        crop_im = crop_im.astype(np.float32)
        norm_im = self.prewhiten(crop_im)   # 1x160x160x3 
        self.feature_extractor.set_tensor(self.input_details[0]['index'], norm_im)
        self.feature_extractor.invoke() 
        emb = self.feature_extractor.get_tensor(self.output_details[0]['index'])
        emb = np.array(emb)
        norm_emb = self.l2_normalize(emb)   # 1xN
        return norm_emb, crop_im

    def prewhiten(self, im):
        mean = np.mean(im)
        std = np.std(im)
        return (im - mean) / std 

    def l2_normalize(self, x):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), 1e-10))
        return output

    def load_image(self, im_name):
        im = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
        crop_im = self.FaceDetector(im)
        return crop_im

    def compare_embeddings(self, feat):
        dist = [] 
        for emb in self.known_embeddings:
            dist += [self.euclidean_dist(feat, emb)]
        min_dist = np.min(dist)
        min_dist_loc = np.argmin(dist) 
        if min_dist < self.recog_th:
            id = self.known_names[min_dist_loc] 
        else:
            id = None 
        return id, min_dist

    def euclidean_dist(self, feat1, feat2):
        return np.linalg.norm(feat1 - feat2)

    def __call__(self, face_crop):
        feat, _ = self.get_embedding_1(face_crop, is_array=True)
        id, _ = self.compare_embeddings(feat)
        return id 



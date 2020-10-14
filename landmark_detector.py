import tensorflow as tf 
import numpy as np 
import cv2
from options import Options 
from face_detection import FaceDetector


class LandmarkDetector():
    def __init__(self):
        self.opts = Options().opts
        self.FaceDetector = FaceDetector() 
        self.interpreter_lm = tf.lite.Interpreter(model_path=self.opts.landmark_model_path)
        self.interpreter_lm.allocate_tensors()
        self.input_details_lm = self.interpreter_lm.get_input_details() 
        self.output_details_lm = self.interpreter_lm.get_output_details() 
        self.input_shape_lm = self.input_details_lm[0]['shape'] 
        self.output_shape_lm = self.output_details_lm[0]['shape']
        self.in_h, self.in_w = self.input_shape_lm[1], self.input_shape_lm[2]

    def __call__(self, face_crop_orig):
        orig_h, orig_w = face_crop_orig.shape[0], face_crop_orig.shape[1]
        face_crop = cv2.cvtColor(face_crop_orig, cv2.COLOR_BGR2RGB)
        face_crop = self.normalize_im(face_crop)
        face_crop_res = cv2.resize(face_crop, (self.in_w, self.in_h))
        face_crop_res = np.reshape(face_crop_res, self.input_details_lm[0]['shape'])
        face_crop_res = face_crop_res.astype(np.float32)
        self.interpreter_lm.set_tensor(self.input_details_lm[0]['index'], face_crop_res)
        self.interpreter_lm.invoke() 
        output_data_lm = self.interpreter_lm.get_tensor(self.output_details_lm[0]['index'])
        landmarks = np.array(output_data_lm)
        circle_im = self.add_circles(face_crop_orig, landmarks, orig_w, orig_h)
        return circle_im

    def unnormalize_im(self, im):
        return (im + 1.0) / 2.0 * 255

    def add_circles(self, face, landmarks, orig_w, orig_h):
        x_vals = landmarks[0, 0, 0, :-1:3] / self.in_w * orig_w 
        y_vals = landmarks[0, 0, 0, 1:-1:3] / self.in_h * orig_h
        new_im = face 
        for x, y in zip(x_vals, y_vals):
            new_im = cv2.circle(new_im, (int(x), int(y)), radius=5, 
                                color=(0, 0, 255),
                                thickness=-1)
        return new_im

    def normalize_im(self, im):
        return 2.0 * im / 255.0 - 1.0

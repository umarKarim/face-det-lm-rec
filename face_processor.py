import cv2
from options import Options 
from face_detection import FaceDetector
from landmark_detector import LandmarkDetector 
from face_recognition import FaceRecognizer 


class FaceProcessor():
    def __init__(self):
        Opts = Options()
        opts = Opts.opts
        self.out_dir = opts.out_dir 
        self.perf_landmark_det = opts.perf_landmark_det 
        self.perf_face_rec = opts.perf_face_rec
        self.FDetector = FaceDetector() 
        if self.perf_landmark_det:
            self.LMDetector = LandmarkDetector() 
        if self.perf_face_rec:
            self.FRecognizer = FaceRecognizer()

    def __call__(self, im_name):
        temp_im = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
        face_crop = self.FDetector(temp_im)
        cv2.imwrite(self.out_dir + 'face_crop.png', cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
        if self.perf_landmark_det:
            lm_im = self.LMDetector(face_crop.copy())
            cv2.imwrite(self.out_dir + 'landmarks.png', cv2.cvtColor(lm_im, cv2.COLOR_RGB2BGR))
        if self.perf_face_rec:
            face_id = self.FRecognizer(face_crop.copy())
            print(face_id)
        print('Process finished')
             

if __name__ == '__main__':
    im_name = Options().opts.test_im_name 
    FaceProcessor().__call__(im_name)
    
import argparse 


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--face_model_path', type=str, default='./models/face_detection_front.tflite')
        parser.add_argument('--landmark_model_path', type=str, default='./models/face_landmark.tflite')
        parser.add_argument('--face_recog_model_path', type=str, default='./models/facenet.tflite')
        parser.add_argument('--anchor_name', type=str, default='anchors.npy')
        parser.add_argument('--face_recog_shape', type=list, default=[160, 160, 3])
        parser.add_argument('--recog_th', type=float, default=2.0)
        parser.add_argument('--face_database', type=str, default='face_database/')
        parser.add_argument('--perf_landmark_det', type=bool, default=True)
        parser.add_argument('--perf_face_rec', type=bool, default=True)
        parser.add_argument('--out_dir', type=str, default='outputs/')
        
        parser.add_argument('--test_im_name', type=str, default='brad_1.jpg')

        self.opts = parser.parse_args() 

    def __call__(self):
        return self.opts
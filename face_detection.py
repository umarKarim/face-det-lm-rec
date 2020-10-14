import numpy as np 
import tensorflow as tf 
from options import Options
import cv2 
import os 
import math


class FaceDetector():
    def __init__(self):
        self.opts = Options().opts
        self.interpreter_det = tf.lite.Interpreter(model_path=self.opts.face_model_path)
        self.interpreter_det.allocate_tensors()
        self.input_details_det = self.interpreter_det.get_input_details()
        self.output_details_det = self.interpreter_det.get_output_details()
        self.input_shape_det = self.input_details_det[0]['shape'] 
        self.output_shape_det = self.output_details_det[1]['shape']
        self.in_h, self.in_w = self.input_shape_det[1], self.input_shape_det[2]

        if not os.path.exists(self.opts.anchor_name):
            self.anchors = self.generate_anchors()
            np.save(self.opts.anchor_name, self.anchors)
        else:
            self.anchors = np.load(self.opts.anchor_name)

    def __call__(self, orig_im):
        orig_h, orig_w = orig_im.shape[0], orig_im.shape[1]
        im_res = cv2.resize(orig_im, (self.in_h, self.in_w))
        im_res = np.array(np.reshape(im_res, self.input_shape_det), dtype=np.float32) / 255.0
        im_norm = self.normalize_im(im_res)
        
        self.interpreter_det.set_tensor(self.input_details_det[0]['index'], im_norm)
        self.interpreter_det.invoke() 
        output_data_box = self.interpreter_det.get_tensor(self.output_details_det[0]['index'])
        output_data_conf = self.interpreter_det.get_tensor(self.output_details_det[1]['index'])
        box, score, loc = self.tensors_to_detections(output_data_box, output_data_conf)

        return self.get_face_crop(box, orig_im, orig_w, orig_h)

    def get_face_crop(self, box, orig_im, orig_w, orig_h):
        eye_r_x = box[4] 
        eye_r_y = box[5]
        eye_l_x = box[6]
        eye_l_y = box[7]
        nose_x = box[8] 
        nose_y = box[9]
        mouth_x = box[10] 
        mouth_y = box[11]
        # aligning the eyes or rotation around z axis
        rot_angle_z = self.get_rotation_z(eye_l_x, eye_r_x, eye_l_y, eye_r_y) 
        rot_mat = cv2.getRotationMatrix2D((eye_l_x, eye_l_y), rot_angle_z * 57.3, 1)
        # rotating and cropping 
        orig_im_rot = cv2.warpAffine(orig_im, rot_mat, (orig_w, orig_h))
        eye_l_coord = np.array([eye_l_x, eye_l_y, 1.0])
        eye_r_coord = np.array([eye_r_x, eye_r_y, 1.0])
        eye_l_x_new = np.dot(np.array(rot_mat[0, :]), eye_l_coord) 
        eye_r_x_new = np.dot(np.array(rot_mat[0, :]), eye_r_coord) 
        # rotation around the y axis 
        rot_angle_y = self.get_rotation_y(eye_l_x, eye_r_x, nose_x, orig_h, orig_w)
        rot_angle_x = self.get_rotation_x()
        transformation_mat = self.get_reproj_mat(rot_angle_y, rot_angle_x, orig_h, orig_w)
        # print(transformation_mat) 
        orig_im_rot = self.apply_reproj(orig_im_rot, transformation_mat, orig_h, orig_w)
        x_max, x_min, y_max, y_min = self.correct_coords(box, orig_w, orig_h)
        face_crop = orig_im_rot[y_min:y_max, x_min:x_max, :]
        return face_crop

    def correct_coords(self, box, orig_w, orig_h):
        x_max, x_min, y_max, y_min = int(box[3] * orig_w), int(box[1] * orig_w), int(box[2] * orig_h), int(box[0] * orig_h)
        y_min, x_min, y_max, x_max = int(box[0] * orig_h), int(box[1] * orig_w), int(box[2] * orig_h), int(box[3] * orig_w)
        face_w, face_h = x_max - x_min, y_max - y_min 
        diff = np.abs(face_w - face_h) / 2.0
        # diff = 0
        if face_h > face_w:
           y_min = y_min + diff 
           y_max = y_max - diff 
        else: 
           x_min = x_min + diff 
           x_max = x_max - diff 
        x_max, y_max = int(x_max + face_w * 0.25), int(y_max + face_h * 0.25)
        x_min, y_min = int(x_min - face_w * 0.25), int(y_min - face_h * 0.25)
        if x_max > orig_w: x_max = orig_w
        if y_max > orig_h: y_max = orig_h 
        if x_min < 0: x_min = 0
        if y_min < 0: y_min = 0
        return x_max, x_min, y_max, y_min

    def get_rotation_y(self, eye_l_x, eye_r_x, nose_x, h, w):
        base = math.sqrt(h * h + w * w) # for simplicity, will be different in real depth case 
        perp = nose_x - (eye_l_x + eye_r_x) / 2.0 
        perp = perp * w
        return math.atan((1.0 * perp) / (1.0 * base)) 

    def get_rotation_x(self):
        return 0 # for the time being 

    def get_reproj_mat(self, angle_y, angle_x, h, w):
        f = math.sqrt(h * h + w * w) 
        dz = f 
        a1 = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 0], [0, 0, 1]])
        r_mat = self.get_rodriguez(angle_y, angle_x) 
        t_mat = np.eye(4) 
        t_mat[2, 3] = dz
        a2 = np.array([[f, 0, w/2, 0], [0, f, h/2, 0], [0, 0, 1, 0]])
        return np.dot(a2, np.dot(t_mat, np.dot(r_mat, a1)))

    def get_rodriguez(self, angle_y, angle_x):
        angle_z = 0.0
        cosz = math.cos(angle_z)
        sinz = math.sin(angle_z) 
        cosx = math.cos(angle_x) 
        sinx = math.sin(angle_x) 
        cosy = math.cos(angle_y) 
        siny = math.sin(angle_y)

        zmat = np.array([[cosz, -sinz, 0.0, 0.0], 
                        [sinz, cosz, 0.0, 0.0], 
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
        ymat = np.array([[cosy, 0.0, siny, 0.0], 
                        [0.0, 1.0, 0.0, 0.0], 
                        [-siny, 0.0, cosy, 0.0],
                        [0.0, 0.0, 0.0, 1.0]]) 
        xmat = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, cosx, -sinx, 0.0], 
                         [0.0, sinx, cosx, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

        rot_mat = xmat @ ymat @ zmat
        rot_mat = np.dot(xmat, np.dot(ymat, zmat))
        return rot_mat 

    def apply_reproj(self, img, mat, h, w):
        return cv2.warpPerspective(img, mat, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)


    def get_rotation_z(self, l_x, r_x, l_y, r_y):
        base = r_x - l_x 
        perp = r_y - l_y 
        return math.atan((1.0 * perp) / (1.0 * base))

    def normalize_im(self, im):
        return 2.0 * im - 1.0

    def calculate_scale(self, min_scale, max_scale, stride_index, num_strides):
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)

    def generate_anchors(self):
        options = {
        "num_layers": 4,
        "min_scale": 0.1484375,
        "max_scale": 0.75,
        "input_size_height": self.in_h,
        "input_size_width": self.in_w,
        "anchor_offset_x": 0.5,
        "anchor_offset_y": 0.5,
        "strides": [8, 16, 16, 16],
        "aspect_ratios": [1.0],
        "reduce_boxes_in_lowest_layer": False,
        "interpolated_scale_aspect_ratio": 1.0,
        "fixed_anchor_size": True,
        }
        strides_size = len(options["strides"])
        assert options["num_layers"] == strides_size

        anchors = []
        layer_id = 0
        while layer_id < strides_size:
            anchor_height = []
            anchor_width = []
            aspect_ratios = []
            scales = []

            # For same strides, we merge the anchors in the same order.
            last_same_stride_layer = layer_id
            while (last_same_stride_layer < strides_size) and \
                (options["strides"][last_same_stride_layer] == options["strides"][layer_id]):
                scale = self.calculate_scale(options["min_scale"],
                                        options["max_scale"],
                                        last_same_stride_layer,
                                        strides_size)

                if last_same_stride_layer == 0 and options["reduce_boxes_in_lowest_layer"]:
                    # For first layer, it can be specified to use predefined anchors.
                    aspect_ratios.append(1.0)
                    aspect_ratios.append(2.0)
                    aspect_ratios.append(0.5)
                    scales.append(0.1)
                    scales.append(scale)
                    scales.append(scale)                
                else:
                    for aspect_ratio in options["aspect_ratios"]:
                        aspect_ratios.append(aspect_ratio)
                        scales.append(scale)

                    if options["interpolated_scale_aspect_ratio"] > 0.0:
                        scale_next = 1.0 if last_same_stride_layer == strides_size - 1 \
                                        else self.calculate_scale(options["min_scale"],
                                                            options["max_scale"],
                                                            last_same_stride_layer + 1,
                                                            strides_size)
                        scales.append(np.sqrt(scale * scale_next))
                        aspect_ratios.append(options["interpolated_scale_aspect_ratio"])

                last_same_stride_layer += 1

            for i in range(len(aspect_ratios)):
                ratio_sqrts = np.sqrt(aspect_ratios[i])
                anchor_height.append(scales[i] / ratio_sqrts)
                anchor_width.append(scales[i] * ratio_sqrts)            
                
            stride = options["strides"][layer_id]
            feature_map_height = int(np.ceil(options["input_size_height"] / stride))
            feature_map_width = int(np.ceil(options["input_size_width"] / stride))

            for y in range(feature_map_height):
                for x in range(feature_map_width):
                    for anchor_id in range(len(anchor_height)):
                        x_center = (x + options["anchor_offset_x"]) / feature_map_width
                        y_center = (y + options["anchor_offset_y"]) / feature_map_height

                        new_anchor = [x_center, y_center, 0, 0]
                        if options["fixed_anchor_size"]:
                            new_anchor[2] = 1.0
                            new_anchor[3] = 1.0
                        else:
                            new_anchor[2] = anchor_width[anchor_id]
                            new_anchor[3] = anchor_height[anchor_id]
                        anchors.append(new_anchor)

            layer_id = last_same_stride_layer
        return anchors

    def sigmoid(self, x): 
        return 1.0 / (1.0 + np.exp(x))

    def decode_boxes(self, raw_boxes, anchors):
        x_scale = 128
        y_scale = 128 
        boxes = np.zeros_like(raw_boxes)
        x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / x_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / y_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(6):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def tensors_to_detections(self, raw_box_tensor, raw_score_tensor):    
        detection_boxes = self.decode_boxes(raw_box_tensor, self.anchors)
        detection_scores = self.sigmoid(raw_score_tensor)
        detection_scores = np.squeeze(detection_scores)
        mask = detection_scores == np.max(detection_scores)
        output_detections = []
        box = detection_boxes[:, np.argmax(raw_score_tensor), :]
        score = detection_scores[np.argmax(raw_score_tensor)]
        return np.squeeze(box), score, np.argmax(detection_scores)
    

    
        
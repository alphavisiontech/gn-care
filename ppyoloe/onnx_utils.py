import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


class PPYOLOEONNXPredictor:
    def __init__(self, model_path, labels_list_path, use_gpu=True, width=640, height=640, threshold=0.5, use_tensorrt=False):
        self.width = width
        self.height = height
        self.threshold = threshold
        # run ONNX session 
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_path, so)
        if use_tensorrt and use_gpu:
            self.net.set_providers(['TensorrtExecutionProvider'])
        elif use_gpu:
            self.net.set_providers(['CUDAExecutionProvider'])
        else:
            self.net.set_providers(['CPUExecutionProvider'])

        # read the labels and store it in a list 
        with open(labels_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.labels = [line.replace('\n', '') for line in lines]

    def load_image(self, img):
        img = cv2.resize(img, (self.width, self.height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') * (1.0 / 255.0)
        img = np.transpose(img, axes=[2, 0, 1])
        img = img.astype(np.float32, copy=False)
        return img
    
    def infer(self, img):
        im_shape = np.array([img.shape[0], img.shape[1]]).astype(np.float32)
        scale_factor = np.array([self.width, self.height], dtype=np.float32) / im_shape
        image = self.load_image(img)
        scale_factor = np.expand_dims(scale_factor, axis=0).astype(np.float32)
        image = np.expand_dims(image, axis=0).astype(np.float32)

        outputs = self.net.run(None, {
            self.net.get_inputs()[0].name: scale_factor,
            self.net.get_inputs()[1].name: image})

        # parse the output and store the [label, bbox, score]
        outs = np.array(outputs[0])
        expect_boxes = (outs[:, 1] > self.threshold) & (outs[:, 0] > -1)
        np_boxes = outs[expect_boxes, :]
        bboxes = []
        scores = []
        for o in np_boxes:
            if int(o[0]) == 0: # person class
                bboxes.append([int(o[2]), int(o[3]), int(o[4]), int(o[5])])
                scores.append(o[1])
            else: 
                continue  # Skip non-person detections

        return bboxes, scores

    @staticmethod
    def draw_box(img, bboxes, scores):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            xmin, ymin, xmax, ymax = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # draw bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=2)
            # write the confidence score 
            draw.text((xmin, ymin - 12), 'Conf: %0.2f' % (score), (0, 0, 255), font=None)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

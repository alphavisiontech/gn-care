import cv2
import numpy as np
import paddle.inference as paddle_infer
from PIL import Image, ImageDraw


class PPYOLOEPredictor:
    def __init__(self, model_dir, use_gpu=True, width=640, height=640, threshold=0.5, use_tensorrt=False):
        self.width = width
        self.height = height
        self.threshold = threshold
        # 创建 config
        config = paddle_infer.Config(f'{model_dir}/model.pdmodel', f'{model_dir}/model.pdiparams')
        if use_gpu:
            config.enable_use_gpu(300, 0)
            if use_tensorrt:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 25,
                    precision_mode=paddle_infer.PrecisionType.Half,
                    use_calib_mode=True)
        else:
            config.set_cpu_math_library_num_threads(4)
            config.enable_mkldnn()
            config.set_mkldnn_cache_capacity(1)
        config.switch_ir_optim(True)
        config.disable_glog_info()
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)

        # 根据 config 创建 predictor
        self.predictor = paddle_infer.create_predictor(config)

        # 获取输入层
        self.image_handle = self.predictor.get_input_handle('image')
        self.scale_factor_handle = self.predictor.get_input_handle('scale_factor')

        # 获取输出的名称
        self.output_names = self.predictor.get_output_names()
        
        labels_list_path = f'../../dataset/label_list.txt' # hard coded path since the file should be exactly this 
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

    def infer(self, im: list):
        scale_factor_list, image_list = [], []
        for i in range(len(im)):
            img_temp = im[i].copy()
            im_shape = np.array([img_temp.shape[0], img_temp.shape[1]]).astype(np.float32)
            scale_factor = np.array([self.width, self.height], dtype=np.float32) / im_shape
            scale_factor_list.append(scale_factor)
            image = self.load_image(img_temp)
            image_list.append(image)
        scale_factor = np.array(scale_factor_list).astype(np.float32)
        image = np.array(image_list).astype(np.float32)

        # preprocess the input
        self.image_handle.reshape(image.shape)
        self.scale_factor_handle.reshape(scale_factor.shape)
        self.image_handle.copy_from_cpu(image)
        self.scale_factor_handle.copy_from_cpu(scale_factor)

        # run the predictor
        self.predictor.run()

        # get the output
        output_handle = self.predictor.get_output_handle(self.output_names[0])
        output_data = output_handle.copy_to_cpu()
        num_handle = self.predictor.get_output_handle(self.output_names[1])
        num_data = num_handle.copy_to_cpu()
        start_num = 0
        bboxes, scores = [], []
        for num in num_data:
            end_num = start_num + num
            output = output_data[start_num:end_num]
            start_num = end_num
            expect_boxes = (output[:, 1] > self.threshold) & (output[:, 0] > -1)
            output = output[expect_boxes, :]
            bbox, score = [], []
            for o in output:
                if int(o[0]) == 0:
                    bbox.append([int(o[2]), int(o[3]), int(o[4]), int(o[5])])
                    score.append(o[1])
                else:
                    continue
            bboxes.append(bbox)
            scores.append(score)
        return bboxes, scores 

    # 对图像进行画框
    @staticmethod
    def draw_box(img, bboxes, scores):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            xmin, ymin, xmax, ymax = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # 画人脸框
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 0, 255), width=2)
            # 绘制文本
            draw.text((xmin, ymin), 'Conf: %0.2f' % (score), (0, 255, 0), font=None)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


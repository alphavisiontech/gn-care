import json
import json
import os
import shutil
import time
from datetime import timedelta

import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader
from tqdm import tqdm
from visualdl import LogWriter

from ppyoloe import SUPPORT_MODEL
from ppyoloe.pytorch_utils.data_utils.reader import CustomDataset, BatchCompose
from ppyoloe.pytorch_utils.metrics.metrics import COCOMetric
from ppyoloe.pytorch_utils.model.utils import get_infer_cfg_and_input_spec
from ppyoloe.pytorch_utils.model.yolo import PPYOLOE_S, PPYOLOE_M, PPYOLOE_L, PPYOLOE_X
from ppyoloe.pytorch_utils.tools.logger import setup_logger
from ppyoloe.pytorch_utils.tools.lr import cosine_decay_with_warmup
from ppyoloe.pytorch_utils.tools.utils import get_coco_model
from ppyoloe.pytorch_utils.tools.utils import get_pretrained_model

logger = setup_logger(__name__)


class PPYOLOETrainer(object):
    def __init__(self,
                 model_type='M',
                 batch_size=8,
                 num_workers=8,
                 num_classes=80,
                 image_dir='dataset/',
                 train_anno_path='dataset/train.json',
                 eval_anno_path='dataset/eval.json',
                 use_gpu=True):
        """PPYOLOE Trainer 
        Args: 
            model_type (str): the selected model type, options are 'S', 'M', 'L', 'X'.
            batch_size (int): the batch size for training or evaluation.
            num_workers (int): the number of workers for data loading.
            num_classes (int): the number of classes for the dataset.
            image_dir (str): the directory containing images.
            train_anno_path (str): the path to the training annotations.
            eval_anno_path (str): the path to the evaluation annotations.
            use_gpu (bool): whether to use GPU for training. If False, CPU will be used.
        """
        if use_gpu:
            assert paddle.is_compiled_with_cuda(), 'GPU is not available!'
            paddle.device.set_device("gpu")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            paddle.device.set_device("cpu")
        self.use_gpu = use_gpu
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.image_dir = image_dir
        self.train_anno_path = train_anno_path
        self.eval_anno_path = eval_anno_path
        assert self.model_type in SUPPORT_MODEL, f'No model: {self.model_type}'
        self.model = None
        self.test_loader = None
        self.metrics = None

    def __setup_dataloader(self,
                           use_random_distort=True,
                           use_random_expand=True,
                           use_random_crop=True,
                           use_random_flip=True,
                           eval_image_size=[640, 640],
                           is_train=False):
        if is_train:
            # Get the data 
            self.train_dataset = CustomDataset(image_dir=self.image_dir,
                                               anno_path=self.train_anno_path,
                                               data_fields=['image', 'gt_bbox', 'gt_class', 'is_crowd'],
                                               mode='train',
                                               use_random_distort=use_random_distort,
                                               use_random_expand=use_random_expand,
                                               use_random_crop=use_random_crop,
                                               use_random_flip=use_random_flip)
            train_batch_sampler = paddle.io.DistributedBatchSampler(self.train_dataset, batch_size=self.batch_size,
                                                                    shuffle=True, drop_last=True)
            collate_fn = BatchCompose()
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           batch_sampler=train_batch_sampler,
                                           collate_fn=collate_fn,
                                           num_workers=self.num_workers)
        # Crate the eval data 
        test_dataset = CustomDataset(image_dir=self.image_dir,
                                     anno_path=self.eval_anno_path,
                                     eval_image_size=eval_image_size,
                                     data_fields=['image'],
                                     mode='eval')
        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers)

    def __setup_model(self, num_epoch=80, learning_rate=1.25e-4, is_train=False):
        # Create the model
        if self.model_type == 'X':
            self.model = PPYOLOE_X(num_classes=self.num_classes)
        elif self.model_type == 'L':
            self.model = PPYOLOE_L(num_classes=self.num_classes)
        elif self.model_type == 'M':
            self.model = PPYOLOE_M(num_classes=self.num_classes)
        elif self.model_type == 'S':
            self.model = PPYOLOE_S(num_classes=self.num_classes)
        else:
            raise Exception(f'No model type exists, model_type: {self.model_type}')
        # print(self.model)
        if paddle.distributed.get_world_size() > 1:
            self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if is_train:
            # Learning rate and weight decay
            self.scheduler = cosine_decay_with_warmup(learning_rate=learning_rate * paddle.distributed.get_world_size(),
                                                      max_epochs=int(num_epoch * 1.2),
                                                      step_per_epoch=len(self.train_loader))
            # Set the optimizer
            self.optimizer = paddle.optimizer.Momentum(parameters=self.model.parameters(),
                                                       learning_rate=self.scheduler,
                                                       momentum=0.9,
                                                       weight_decay=paddle.regularizer.L2Decay(5e-4))

    def __load_pretrained(self, pretrained_model=None):
        # Load the pretrained model
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
            assert os.path.exists(pretrained_model), f"{pretrained_model} No such model exists!"
            model_dict = self.model.state_dict()
            model_state_dict = paddle.load(pretrained_model)
            # Filter the model state dict (non existent parameters or shape mismatch)
            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if list(weight.shape) != list(model_state_dict[name].shape):
                        logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                       format(name, list(model_state_dict[name].shape), list(weight.shape)))
                        model_state_dict.pop(name, None)
                else:
                    logger.warning('Lack weight: {}'.format(name))
            self.model.set_state_dict(model_state_dict)
            logger.info('Successfully loaded pretrained model: {}'.format(pretrained_model))
        else:
            # Load the official pretrained model
            pretrained_path = get_pretrained_model(model_type=self.model_type)
            self.model.set_state_dict(paddle.load(pretrained_path))
            logger.info('Successfully loaded pretrained model: {}'.format(pretrained_path))

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_mAP = 0
        last_model_dir = os.path.join(save_model_path, f'PPYOLOE_{self.model_type}', 'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pdparams'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pdopt'))):
            # Automatically get the latest saved model
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pdparams')), "Model parameter file does not exist!"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pdopt')), "Optimizer parameter file does not exist!"
            self.model.set_state_dict(paddle.load(os.path.join(resume_model, 'model.pdparams')))
            self.optimizer.set_state_dict(paddle.load(os.path.join(resume_model, 'optimizer.pdopt')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                best_mAP = json_data['mAP']
            logger.info('Successfully restore checkpoint: {}'.format(resume_model))
        return last_epoch, best_mAP

    # Save checkpoint
    def __save_checkpoint(self, save_model_path, epoch_id, mAP=0, best_model=False):
        if best_model:
            model_path = os.path.join(save_model_path, f'PPYOLOE_{self.model_type}', 'best_model')
        else:
            model_path = os.path.join(save_model_path, f'PPYOLOE_{self.model_type}', 'epoch_{}'.format(epoch_id))
        os.makedirs(model_path, exist_ok=True)
        try:
            paddle.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
            paddle.save(self.model.state_dict(), os.path.join(model_path, 'model.pdparams'))
        except Exception as e:
            logger.error(f'Error occurred while saving model: {e}')
            return
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            f.write('{"last_epoch": %d, "mAP": %f}' % (epoch_id, mAP))
        if not best_model:
            last_model_path = os.path.join(save_model_path, f'PPYOLOE_{self.model_type}', 'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # Delete the old model
            old_model_path = os.path.join(save_model_path, f'PPYOLOE_{self.model_type}',
                                          'epoch_{}'.format(epoch_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('Successfully saved model: {}'.format(model_path))

    def __train_epoch(self, max_epoch, epoch_id, log_interval, local_rank, writer):
        train_times, loss_sum = [], []
        start = time.time()
        sum_batch = len(self.train_loader) * max_epoch
        for batch_id, data in enumerate(self.train_loader()):
            data['epoch_id'] = epoch_id
            output = self.model(data)
            # Loss calcuation and backpropagation
            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
            loss_sum.append(loss.numpy()[0])
            train_times.append((time.time() - start) * 1000)

            # Log training information
            if batch_id % log_interval == 0 and local_rank == 0:
                # Calculate training speed
                train_speed = self.batch_size / (sum(train_times) / len(train_times) / 1000)
                # Calculate remaining time
                eta_sec = (sum(train_times) / len(train_times)) * (
                        sum_batch - (epoch_id - 1) * len(self.train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(f'Train epoch: [{epoch_id}/{max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {sum(loss_sum) / len(loss_sum):.5f}, '
                            f'learning rate: {self.scheduler.get_lr():>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', sum(loss_sum) / len(loss_sum), self.train_step)
                train_times = []
            self.scheduler.step()
            # Record learning rate
            writer.add_scalar('Train/lr', self.scheduler.get_lr(), self.train_step)
            start = time.time()

    def train(self,
              num_epoch=80,
              learning_rate=1.25e-4,
              log_interval=100,
              use_random_distort=True,
              use_random_expand=True,
              use_random_crop=True,
              use_random_flip=True,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None):
        """
        Training the PPYOLOE model
        
        Args:
            num_epoch (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
            log_interval (int): Interval for logging training information.
            use_random_distort (bool): Whether to use random distortion during training.
            use_random_expand (bool): Whether to use random expansion during training.
            use_random_crop (bool): Whether to use random cropping during training.
            use_random_flip (bool): Whether to use random flipping during training.
            save_model_path (str): Path to save the trained model.
            resume_model (str): Path to resume training from a saved model.
            pretrained_model (str): Path to a pretrained model to load weights from.
        """
        paddle.seed(1000)
        # Train on how many GPUs
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        writer = None
        if local_rank == 0:
            # Logger
            writer = LogWriter(logdir='log')

        if nranks > 1 and self.use_gpu:
            # Initialize distributed training
            strategy = fleet.DistributedStrategy()
            fleet.init(is_collective=True, strategy=strategy)

        # Setup dataset
        self.__setup_dataloader(use_random_distort=use_random_distort,
                                use_random_expand=use_random_expand,
                                use_random_crop=use_random_crop,
                                use_random_flip=use_random_flip,
                                is_train=True)
        # Setup model
        self.__setup_model(num_epoch=num_epoch, learning_rate=learning_rate, is_train=True)
        # Evaluation metric
        self.metrics = COCOMetric(anno_file=self.eval_anno_path)
        # Multi-GPU training
        if nranks > 1 and self.use_gpu:
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
            self.model = fleet.distributed_model(self.model)
        logger.info(f'Training data: {len(self.train_dataset)}')

        self.__load_pretrained(pretrained_model=pretrained_model)
        # Load the checkpoint if exists
        last_epoch, best_mAP = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        test_step, self.train_step = 0, 0
        last_epoch += 1
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_lr(), last_epoch)
        # Start training
        for epoch_id in range(last_epoch, num_epoch):
            epoch_id += 1
            start_epoch = time.time()
            # Train the model for one epoch
            self.__train_epoch(max_epoch=num_epoch, epoch_id=epoch_id, log_interval=log_interval, local_rank=local_rank, writer=writer)
            # Multi-GPU training only uses one process to perform evaluation and save the model
            if local_rank == 0:
                logger.info('=' * 70)
                mAP = self.evaluate(resume_model=None)[0]
                # Save the best model
                if mAP >= best_mAP:
                    best_mAP = mAP
                    self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, mAP=mAP,
                                           best_model=True)
                logger.info('Test epoch: {}, time/epoch: {}, best_mAP: {:.5f}, mAP: {:.5f}'.format(
                    epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), best_mAP, mAP))
                logger.info('=' * 70)
                writer.add_scalar('Test/mAP', mAP, test_step)
                test_step += 1
                self.model.train()
                # Save checkpoint for the current epoch
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, mAP=mAP)

    def evaluate(self, image_size='640,640', resume_model='models/PPYOLOE_M/best_model/'):
        """
        Evaluate the trained model on the validation dataset.

        Args:
            image_size (str): The input size of the images for evaluation, e.g., '640,640'.
            resume_model (str): The path to the model to resume from.
        """
        if self.metrics is None:
            self.metrics = COCOMetric(anno_file=self.eval_anno_path)
        if self.test_loader is None:
            eval_image_size = [int(s) for s in image_size.split(',')]
            self.__setup_dataloader(eval_image_size=eval_image_size)
        if self.model is None:
            self.__setup_model()
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pdparams')
            assert os.path.exists(resume_model), f"{resume_model} No such model exists!"
            model_state_dict = paddle.load(resume_model)
            self.model.set_state_dict(model_state_dict)
            logger.info(f'Successfully load the model: {resume_model}')
        self.model.eval()
        if isinstance(self.model, paddle.DataParallel):
            eval_model = self.model._layers
        else:
            eval_model = self.model

        with paddle.no_grad():
            for batch_id, data in enumerate(tqdm(self.test_loader())):
                outputs = eval_model(data)
                self.metrics.update(inputs=data, outputs=outputs)
        mAP = self.metrics.accumulate()
        self.metrics.reset()
        self.model.train()
        return mAP

    def export(self, image_shape='3,640,640', save_model_path='models/', resume_model='models/PPYOLOE_M/best_model/'):
        """
        Export the model 

        Args: 
            image_shape (str): The input shape of the model, e.g., '3,640,640'.
            save_model_path (str): The path to save the exported model.
            resume_model (str): The path to the model to resume from.
        """
        # Setup model
        self.__setup_model()
        # Load pretrained model
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pdparams')
            assert os.path.exists(resume_model), f"{resume_model} No such model exists!"
            model_state_dict = paddle.load(resume_model)
            self.model.set_state_dict(model_state_dict)
            logger.info(f'Successfully load the model: {resume_model}')
        else:
            # Load official pretrained model
            pretrained_path = get_coco_model(model_type=self.model_type)
            self.model.set_state_dict(paddle.load(pretrained_path))
            logger.info(f'Successfully load the pretrained model: {pretrained_path}')
        self.model.eval()
        # Get static model
        image_shape = [int(i) for i in image_shape.split(',')]
        static_model, pruned_input_spec = get_infer_cfg_and_input_spec(model=self.model, image_shape=image_shape)
        infer_model_dir = os.path.join(save_model_path, f'PPYOLOE_{self.model_type.upper()}', 'infer')
        paddle.jit.save(static_model, os.path.join(infer_model_dir, 'model'), input_spec=pruned_input_spec)
        logger.info(f'Exported model saved in: {infer_model_dir}')

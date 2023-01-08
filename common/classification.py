#!/usr/bin/env python3
import logging, cv2
from time import perf_counter
import numpy as np
from openvino.inference_engine import IECore
from .model import Model
from .pipelines import get_user_config, AsyncPipeline
# from .performance_metrics import PerformanceMetrics
from .utils import load_labels, resize_image, OutputTransform

class Classification:
    def __init__(self, dict):
        self.next_frame_id = 0
        self.next_frame_id_to_show = 0
        # self.metrics = PerformanceMetrics()
        self.dict = dict

    def load_model(self):
        # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
        logging.info('Initializing Inference Engine...')
        ie = IECore()
        # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
        self.model = ClassificationModel(ie, self.dict['model_path'], 
                                        labels=self.dict['label_path'])
        logging.info('Reading the network: {}'.format(self.dict['model_path']))
        logging.info('Loading network...')
        # Get device relative info for inference 
        plugin_config = get_user_config( self.dict['device'], self.dict["num_streams"], self.dict["num_threads"])
        # Initialize Pipeline(for inference)
        self.detector_pipeline = AsyncPipeline(ie, self.model, plugin_config,
                                                device=self.dict['device'], max_num_requests=self.dict["num_infer_requests"])
        # ---------------------------Step 3. Create detection words of color---------------
        # color = tuple(np.random.choice(range(256), size=3).tolist())
        palette = [[255,0,0]]

        return palette

    def inference(self, frame):
        # Label length
        num_of_classes = len(self.model.labels)
        # Check pipeline is ready & setting output_shape & inference
        if self.detector_pipeline.is_ready():
            start_time = perf_counter()
            if frame is None:
                if self.next_frame_id == 0:
                    raise ValueError("Can't read an image from the input")
                raise ValueError("Can't read an image")
            if self.next_frame_id == 0:
                # Compute rate from setting output shape and input images shape 
                self.output_transform = OutputTransform(frame.shape[:2], None)     
            # Submit for inference
            self.detector_pipeline.submit_data(frame, self.next_frame_id, {'frame': frame, 'start_time': start_time})
            self.next_frame_id += 1
        else:
            # Wait for empty request
            self.detector_pipeline.await_any()
        if self.detector_pipeline.callback_exceptions:
            raise self.detector_pipeline.callback_exceptions[0]

        results = self.detector_pipeline.get_result(self.next_frame_id_to_show)
        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']
            # self.metrics.update(start_time, frame)
            self.next_frame_id_to_show += 1
            info = {"frame":frame,
                        "output_transform":self.output_transform}
            total_bbox = []
            for detection in objects:
                # Change a shape of a numpy.ndarray with results to get another one with one dimension
                probs = objects[detection].reshape(num_of_classes)
                # Get an array of args.number_top class IDs in descending order of probability
                top_n_idexes = np.argmax(probs)
                score = probs[top_n_idexes]
                if score < 0.5:
                    top_n_idexes = -1
                    score = 1.0
                total_bbox.append({'xmin':15, 
                                    'ymin':40, 
                                    'xmax':30, 
                                    'ymax':30, 
                                    'det_label':self.model.labels[top_n_idexes], 
                                    'score': score,
                                    'id': top_n_idexes})
                logging.info('{:^9} | {:10f} '
                            .format(self.model.labels[top_n_idexes], score))
            info.update({'detections':total_bbox}) 
            # print("next_frame_id:{}, next_frame_id_to_show:{}".format(self.next_frame_id, self.next_frame_id_to_show))
            return info

class ClassificationModel(Model):
    def __init__(self, ie, model_path, labels=None):
        super().__init__(ie, model_path)

        if isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            self.labels = load_labels(labels) if labels else None

        self.resize_image = resize_image
        
        assert len(self.net.input_info) == 1, "Expected 1 input blob"
        self.image_blob_name = next(iter(self.net.input_info))
        if self.net.input_info[self.image_blob_name].input_data.shape[1] == 3:
            self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
            self.nchw_shape = True
        else:
            self.n, self.h, self.w, self.c = self.net.input_info[self.image_blob_name].input_data.shape
            self.nchw_shape = False

    def preprocess(self, inputs):
        image = cv2.cvtColor(
            src=inputs,
            code=cv2.COLOR_BGR2RGB,
        )
        # logging.warning('Image is resized from {} to {}'.format(image.shape[:-1],(self.h,self.w)))
        resized_image = self.resize_image(image, (self.w, self.h))
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}
        if self.nchw_shape:
            resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))

        else:
            resized_image = resized_image.reshape((self.n, self.h, self.w, self.c))

        dict_inputs = {self.image_blob_name: resized_image}

        return dict_inputs, meta
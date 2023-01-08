#!/usr/bin/env python3
import cv2, os, logging
import numpy as np
from time import perf_counter
from pathlib import Path

from .model import Model
from .pipelines import get_user_config, AsyncPipeline
from .performance_metrics import PerformanceMetrics
from .utils import load_txt, OutputTransform
from .draw import u2net_postprocessing
from openvino.inference_engine import IECore

class Segmentation:
    def __init__(self, dict):
        self.next_frame_id = 0
        self.next_frame_id_to_show = 0
        self.metrics = PerformanceMetrics()
        self.dict = dict

    def load_model(self):
        # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
        logging.info('Initializing Inference Engine...')
        ie = IECore()
        # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
        self.model = get_model(ie, self.dict)
        logging.info('Reading the network: {}'.format(self.dict['model_path']))
        logging.info('Loading network...')
        # Get device relative info for inference 
        plugin_config = get_user_config( self.dict['device'], self.dict["num_streams"], self.dict["num_threads"])
        # Initialize Pipeline(for inference)
        self.detector_pipeline = AsyncPipeline(ie, self.model, plugin_config,
                                        device = self.dict['device'], max_num_requests = self.dict["num_infer_requests"])
        # ---------------------------Step 3. Create detection words of color---------------
        if self.dict['architecture_type'] == 'segmentation':
            palette = SegmentationVisualizer(self.dict)
        else:
            palette = SaliencyMapVisualizer()

        return palette

    def inference(self, frame):
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
        # Process all completed requests
        results = self.detector_pipeline.get_result(self.next_frame_id_to_show)

        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']
            # self.metrics.update(start_time, frame)
            self.next_frame_id_to_show += 1
            info = {"frame":frame,
                    "output_transform":self.output_transform,
                    'only_masks':self.dict['only_masks'],
                    'detections':[{'objects':objects}]}
            # print("S-next_frame_id:{}, S-next_frame_id_to_show:{}".format(self.next_frame_id, self.next_frame_id_to_show))
            return info

def get_model(ie, config):
    if config['architecture_type'] == 'segmentation':
        return SegmentationModel(ie, config['model_path'])
    if config['architecture_type'] == 'salient_object_detection':
        return SalientObjectDetectionModel(ie, config['model_path'])

class SegmentationModel(Model):
    def __init__(self, ie, model_path):
        super().__init__(ie, model_path)

        self.input_blob_name = self.prepare_inputs()
        self.out_blob_name = self.prepare_outputs()
        self.labels = None
    
    def prepare_inputs(self):
        if len(self.net.input_info) != 1:
            raise RuntimeError("Demo supports topologies only with 1 input")
        blob_name = next(iter(self.net.input_info)) # get next iter to value
        blob = self.net.input_info[blob_name]
        blob.precision = "U8"
        blob.layout = "NCHW"

        input_size = blob.input_data.shape
        if len(input_size) == 4 and input_size[1] == 3:
            self.n, self.c, self.h, self.w = input_size
        else:
            raise RuntimeError("3-channel 4-dimensional model's input is expected")

        return blob_name

    def prepare_outputs(self):
        # if len(self.net.outputs) != 1:
        #     raise RuntimeError("Demo supports topologies only with 1 output")
        blob_name = next(iter(self.net.outputs)) # get next iter to value
        blob = self.net.outputs["1959"]

        out_size = blob.shape
        if len(out_size) == 3:
            self.out_channels = 0
        elif len(out_size) == 4:
            self.out_channels = out_size[1]
        else:
            raise Exception("Unexpected output blob shape {}. Only 4D and 3D output blobs are supported".format(out_size))

        return blob_name

    def preprocess(self, inputs):
        image = cv2.cvtColor(
            src=inputs,
            code=cv2.COLOR_BGR2RGB,
        )
        # image = inputs
        resized_image = cv2.resize(image, (self.w, self.h))
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))
        dict_inputs = {self.input_blob_name: resized_image}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        result = np.rint(
            cv2.resize(src=np.squeeze(outputs[self.out_blob_name]), dsize=(meta['original_shape'][1], meta['original_shape'][0]))
        ).astype(np.uint8)
        # predictions = outputs[self.out_blob_name].squeeze() # Remove axes of length one from a
        # input_image_height = meta['original_shape'][0]
        # input_image_width = meta['original_shape'][1]

        # if self.out_channels < 2: # assume the output is already ArgMax'ed
        #     result = predictions.astype(np.uint8)
        # else:
        #     result = np.argmax(predictions, axis=0).astype(np.uint8)

        # result = cv2.resize(result, (input_image_width, input_image_height), 0, 0, interpolation=cv2.INTER_NEAREST)
        return result

class SalientObjectDetectionModel(SegmentationModel):

    def postprocess(self, outputs, meta):
        input_image_height = meta['original_shape'][0]
        input_image_width = meta['original_shape'][1]
        result = outputs[self.out_blob_name].squeeze()
        result = 1/(1 + np.exp(-result))
        result = cv2.resize(result, (input_image_width, input_image_height), 0, 0, interpolation=cv2.INTER_NEAREST)
        return result

class SegmentationVisualizer:
    pascal_voc_palette = [
        (0,   0,   0),
        (128, 0,   0),
        (0,   128, 0),
        (128, 128, 0),
        (0,   0,   128),
        (128, 0,   128),
        (0,   128, 128),
        (128, 128, 128),
        (64,  0,   0),
        (192, 0,   0),
        (64,  128, 0),
        (192, 128, 0),
        (64,  0,   128),
        (192, 0,   128),
        (64,  128, 128),
        (192, 128, 128),
        (0,   64,  0),
        (128, 64,  0),
        (0,   192, 0),
        (128, 192, 0),
        (0,   64,  128)
    ]

    def __init__(self, Dict, colors_path=None):
        if colors_path:
            self.color_palette = self.get_palette_from_file(colors_path)
        else:
            self.color_palette = self.pascal_voc_palette
        self.color_map = self.create_color_map()
        self.get_palette(Dict)

    def get_palette_from_file(self, colors_path):
        with open(colors_path, 'r') as file:
            colors = []
            for line in file.readlines():
                values = line[line.index('(')+1:line.index(')')].split(',')
                colors.append([int(v.strip()) for v in values])
            return colors

    def create_color_map(self):
        classes = np.array(self.color_palette, dtype=np.uint8)[:, ::-1] # RGB to BGR
        color_map = np.zeros((256, 1, 3), dtype=np.uint8)
        classes_num = len(classes)
        color_map[:classes_num, 0, :] = classes
        color_map[classes_num:, 0, :] = np.random.uniform(0, 255, size=(256-classes_num, 3))
        return color_map

    def apply_color_map(self, input):
        input_3d = cv2.merge([input, input, input])
        return cv2.LUT(input_3d, self.color_map) # 使用查找表中的值填充输出数组

    def get_palette(self, config):
        # init
        palette, content = list(), list()
        # get labels
        if not ('label_path' in config.keys()):
            mas = "Error configuration file, can't find `label_path`"
            logging.error(mas)
            raise Exception(mas)
        label_path = config['label_path']
        labels = load_txt(label_path)
        # parse the path
        name, ext = os.path.splitext(label_path)
        output_palette_path = "{}_colormap{}".format(name, ext)
        # update palette and the content of colormap
        logging.info("Get colormap ...")
        for index, label in enumerate(labels):
            color = self.color_map[index][0]                                       # get random color
            palette.append(color)                                               # get palette's color list
            content.append('{label}: {color}'.format(label=label, color=tuple(color)))  # setup content
        # write map table into colormap
        logging.info("Write colormap into `{}`".format(output_palette_path))
        with open(output_palette_path, 'w') as f:
            f.write('\n'.join(content))

class SaliencyMapVisualizer:
    def apply_color_map(self, input):
        saliency_map = (input * 255.0).astype(np.uint8)
        saliency_map = cv2.merge([saliency_map, saliency_map, saliency_map])
        return saliency_map

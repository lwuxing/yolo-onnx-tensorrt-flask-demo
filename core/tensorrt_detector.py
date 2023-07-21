"""
This module implements the Tensorrt inference class.
"""

import ctypes

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # This is needed for initializing CUDA driver

from utils import BBoxVisualization

def _preprocess_trt(img, shape=(640, 640), letter_box=False):
    """Preprocess an image before TensorRT inferencing.

    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = shape[0], shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((shape[0], shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (shape[1], shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    img = np.ascontiguousarray(img)
    return img


def _nms_boxes(detections, scores, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.

    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]
    y_coord = detections[:, 1]
    width = detections[:, 2]
    height = detections[:, 3]

    areas = width * height
    ordered = scores.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)
        iou = intersection / union
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape

    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    # Clip bounding xyxy bounding boxes to image shape (height, width)
    coords[:, 0].clip(0, img0_shape[1])  # x1
    coords[:, 1].clip(0, img0_shape[0])  # y1
    coords[:, 2].clip(0, img0_shape[1])  # x2
    coords[:, 3].clip(0, img0_shape[0])  # y2

    return coords


def _postprocess_trt(predictions, img1_shape, img0_shape, conf_th, nms_threshold, letter_box=False):
    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_th, :]
    scores = scores[scores > conf_th]

    if len(scores) == 0:
        return np.array([]), np.array([]), np.array([])

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Extract boxes from predictions
    # Convert boxes to xyxy format
    # Scale boxes to original image dimensions
    boxes = predictions[:, :4]
    boxes = xywh2xyxy(boxes)
    boxes = scale_coords(img1_shape, boxes, img0_shape)

    indices = _nms_boxes(boxes, scores, nms_threshold)
    return boxes[indices], scores[indices], class_ids[indices]
 

def get_input_shape(engine):
    """Get input shape of the TensorRT YOLO engine."""
    binding = engine[0]
    assert engine.binding_is_input(binding)
    binding_dims = engine.get_binding_shape(binding)
    if len(binding_dims) == 4:
        return tuple(binding_dims[2:])
    elif len(binding_dims) == 3:
        return tuple(binding_dims[1:])
    else:
        raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))

class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.device
        del self.host   


class TrtModel:
    """TrtSSD class encapsulates things needed to run TRT SSD."""

    def __load_plugins(self):
        # if trt.__version__[0] < '7':
        #     ctypes.CDLL("ssd/libflattenconcat.so")
        # trt.init_libnvinfer_plugins(self.trt_logger, '')
        pass

    def __load_engine(self):
        with open(self.model, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __allocate_buffers(self):
        # host_inputs, host_outputs, cuda_inputs, cuda_outputs = [], [], [], []
        inputs, outputs, bindings = [], [], []
        device_mem_list = []

        for binding in self.engine:

            binding_dims = self.engine.get_binding_shape(binding)
            if len(binding_dims) == 4:
                # explicit batch case (TensorRT 7+)
                size = trt.volume(binding_dims)
            elif len(binding_dims) == 3:
                # implicit batch case (TensorRT 6 or older)
                size = trt.volume(binding_dims) * self.engine.max_batch_size
            else:
                raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            # issue: https://blog.csdn.net/weixin_42115033/article/details/82466342
            device_mem_list.append(cuda_mem)
            # int(cuda_mem) 表示在cuda上申请的GPU显存地址
            bindings.append(int(cuda_mem))

            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                # host_inputs.append(host_mem)
                # cuda_inputs.append(cuda_mem)
                inputs.append(HostDeviceMem(host_mem, cuda_mem))
            else:
                # host_outputs.append(host_mem)
                # cuda_outputs.append(cuda_mem)
                 outputs.append(HostDeviceMem(host_mem, cuda_mem))
        # return host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings
        # del device_mem_list

        return inputs, outputs, bindings

    def __init__(self, model, input_shape=None, letter_box=True, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.letter_box = letter_box
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.__load_plugins()
        self.engine = self.__load_engine()
        self.input_shape = input_shape or get_input_shape(self.engine)
        try:
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.inputs, self.outputs, self.bindings = self.__allocate_buffers()
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    # def __del__(self):
    #     """Free CUDA memories and context."""

    #     # del operator will cause an error, why?
    #     del self.outputs
    #     del self.inputs
    #     del self.stream

    def __call__(self, img, conf_th=0.3):
        """Detect objects in the input image."""
        img_resized = _preprocess_trt(img, self.input_shape, self.letter_box)
        # Set host input to the image. 
        np.copyto(self.inputs[0].host, img_resized.ravel())

        # copy the input to the GPU before executing.
        if self.cuda_ctx:
            self.cuda_ctx.push()
        outputs = do_inference_v2(context=self.context, 
                                  bindings=self.bindings, 
                                  inputs=self.inputs, 
                                  outputs=self.outputs, 
                                  stream=self.stream)
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        outputs = outputs[0].reshape(-1, 8400).T
        boxes, scores, classes = _postprocess_trt(outputs, self.input_shape, img.shape, conf_th,
                                                  nms_threshold=0.5, 
                                                  letter_box=self.letter_box)

        # clip x1, y1, x2, y2 within original image
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img.shape[1]-1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img.shape[0]-1)

        pred_reformat = []
        for _, det in enumerate(zip(boxes, scores, classes)):

            pred_reformat.append(
                {
                    "name": int(det[-1]),
                    "score": round(float(np.float32(det[1])), 3),
                    "bbox":[round(float(val), 2) for val in det[0].tolist()]

                })
        return pred_reformat

def do_inference_v2(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


if __name__ == "__main__":
    import time
    engine_path = './yolov5.engine'
    image_path = 'fall_0.jpg'

    model = TrtModel(engine_path, cuda_ctx=pycuda.autoinit.context)
    img = cv2.imread(image_path)

    tic = time.time()
    boxes, scores, classes = model.detect(img)
    print(time.time() - tic)

    print(boxes, scores, classes)
    vis = BBoxVisualization(dict((i, clas) for i, clas in enumerate(["down", "person", "10+", "dog"])))
    im = vis.draw_bboxes(img, boxes, scores, classes)
    cv2.imwrite('test1.jpg', im)
    # model.release()
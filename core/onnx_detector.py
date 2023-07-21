import sys, os
import cv2
sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0])

import numpy as np
from utils import *

np.set_printoptions(precision=4)


class Detector:
    def __init__(self, model_path, inference="onnx", imgsz=640):

        self.model_path = model_path
        self.inference = inference
        self.imgsz = imgsz
        self.class_names = {0:"fire", 1:"fog"}
        self.conf_threshold = 0.6
        self.iou_threshold = 0.5
        if self.inference == 'onnx':
            import onnxruntime as ort
            providers = ort.get_available_providers()

            self.session = ort.InferenceSession(self.model_path, providers=providers)

            model_inputs = self.session.get_inputs()
            self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

            # self.input_shape = model_inputs[0].shape
            # self.input_height = self.input_shape[2]
            # self.input_width = self.input_shape[3]

            model_outputs = self.session.get_outputs()
            self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        else:
            raise NotImplementedError('not implemented type')

    def preprocess(self, image_ori, imgsz):
        # ---preprocess image for detection
        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        image = letterbox(image, imgsz)[0]
        image = image.astype(np.float32)
        image = image / 255.0  # 0 - 255 to 0.0 - 1.0
        image = np.transpose(image, [2, 0, 1])  # HWC to CHW, BGR to RGB
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image

    def poseprocess(self, outputs, img1_shape, img0_shape):
        predictions = np.squeeze(outputs[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

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

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def __call__(self, image_ori):

        image = self.preprocess(image_ori, self.imgsz)

        if self.inference == "onnx":
            # is list, [shape(1, 8, 8400)]
            outputs = self.session.run(self.output_names, {self.input_names[0]: image})

        else:
            raise NotImplementedError('')

        self.boxes, self.scores, self.class_ids = self.poseprocess(outputs, image.shape[2:], image_ori.shape)
        pred_reformat = []
        try:
            for _, det in enumerate(zip(self.boxes, self.scores, self.class_ids)):

                pred_reformat.append(
                    {
                        "name": int(det[-1]),
                        "score": round(float(np.float32(det[1])), 3),
                        "bbox":[round(float(val), 2) for val in det[0].tolist()]

                    })
        except:
            import traceback
            print(traceback.format_exc())

        return pred_reformat

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return self._do_draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def _do_draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.3):
        mask_img = image.copy()
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        # Draw bounding boxes and labels of detections
        rng = np.random.default_rng(3)
        self.colors = rng.uniform(0, 255, size=(len(self.class_names), 3))
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = self.colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Draw fill rectangle in mask image
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            label = self.class_names[class_id]
            caption = f'{label} {"%.2f" % score}'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.rectangle(mask_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


if __name__ == "__main__":
    model = Detector('../weights/best.onnx')
    img = cv2.imread('../uploads/fall_0.jpg')

    print(model(img))
    img = model.draw_detections(img)
    cv2.imwrite('../uploads/test1.jpg', img)

import numpy as np
import onnxruntime as ort
import cv2
from typing import List, Tuple


class ObjectDetector:
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45, use_gpu: bool = False):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        providers = ['CPUExecutionProvider']
        if use_gpu:
            providers.insert(0, 'CUDAExecutionProvider')

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            self.input_size = (input_shape[2], input_shape[3]) if len(input_shape) == 4 else (640, 640)
            self.model_loaded = True
        except Exception as e:
            print(f"[WARN] Failed to load detection model: {e}. Using mock detector.")
            self.model_loaded = False
            self.input_size = (640, 640)

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        original_h, original_w = frame.shape[:2]
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, (original_w, original_h)

    def _postprocess(self, outputs: np.ndarray, original_size: Tuple[int, int]) -> List[dict]:
        original_w, original_h = original_size
        input_w, input_h = self.input_size

        predictions = np.squeeze(outputs).T

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            score = class_scores[class_id]

            if score > self.conf_threshold:
                cx, cy, w, h = pred[:4]

                x1 = (cx - w / 2) / input_w * original_w
                y1 = (cy - h / 2) / input_h * original_h
                x2 = (cx + w / 2) / input_w * original_w
                y2 = (cy + h / 2) / input_h * original_h

                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))
                class_ids.append(int(class_id))

        if len(boxes) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        results = []
        for i in indices.flatten() if len(indices) > 0 else []:
            x1, y1, x2, y2 = boxes[i]
            class_id = class_ids[i]
            results.append({
                "class": self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else f"class_{class_id}",
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": scores[i]
            })

        return results

    def detect(self, frame: np.ndarray) -> List[dict]:
        if not self.model_loaded:
            return self._mock_detect(frame)

        input_data, original_size = self._preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return self._postprocess(outputs[0], original_size)

    def _mock_detect(self, frame: np.ndarray) -> List[dict]:
        h, w = frame.shape[:2]
        frame_hash = hash(str(frame.shape) + str(frame.mean()))
        mock_objects = [
            ("person", [0.1, 0.2, 0.3, 0.8]),
            ("car", [0.4, 0.5, 0.9, 0.8]),
            ("chair", [0.2, 0.6, 0.4, 0.9]),
            ("tv", [0.5, 0.1, 0.8, 0.4]),
            ("bottle", [0.7, 0.5, 0.75, 0.7]),
        ]
        count = frame_hash % 4
        results = []
        for i in range(count + 1):
            obj_class, bbox_rel = mock_objects[(frame_hash + i) % len(mock_objects)]
            bbox = [
                bbox_rel[0] * w, bbox_rel[1] * h,
                bbox_rel[2] * w, bbox_rel[3] * h
            ]
            results.append({
                "class": obj_class,
                "bbox": bbox,
                "confidence": 0.7 + (frame_hash % 30) / 100
            })
        return results

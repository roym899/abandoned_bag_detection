# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from abandoned_bag_heuristic import SimpleTracker

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
import detectron2.utils.video_visualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

SAVE_PREDICTIONS = False
SAVED_PREDICTIONS = []

import pickle


def draw_instance_predictions(visualizer, frame, predictions, tracker):
    """
    Draw instance-level prediction results on an image.

    Args:
        frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

    Returns:
        output (VisImage): image object with visualizations.
    """
    frame_visualizer = Visualizer(frame, visualizer.metadata)
    num_instances = len(predictions)
    if num_instances == 0:
        return frame_visualizer.output

    boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    if predictions.has("pred_masks"):
        masks = predictions.pred_masks
        # mask IOU is not yet enabled
        # masks_rles = mask_util.encode(np.asarray(masks.permute(1, 2, 0), order="F"))
        # assert len(masks_rles) == num_instances
    else:
        masks = None

    detected = [
        detectron2.utils.video_visualizer._DetectedInstance(classes[i], boxes[i], mask_rle=None, color=None, ttl=8)
        for i in range(num_instances)
    ]
    colors = visualizer._assign_colors(detected)

    labels = detectron2.utils.video_visualizer._create_text_labels(classes, scores,
                                                                   visualizer.metadata.get("thing_classes", None))

    if visualizer._instance_mode == ColorMode.IMAGE_BW:
        # any() returns uint8 tensor
        frame_visualizer.output.img = frame_visualizer._create_grayscale_image(
            (masks.any(dim=0) > 0).numpy() if masks is not None else None
        )
        alpha = 0.3
    else:
        alpha = 0.5

    frame_visualizer.overlay_instances(
        boxes=None if masks is not None else boxes,  # boxes are a bit distracting
        masks=masks,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )

    for bag_id in tracker.prev_frame_ids['bags']:
        bag_center = tracker.all_centers['bags'][bag_id]
        if bag_id in tracker.bag_person_association:
            person_id = tracker.bag_person_association[bag_id]
            if person_id is not None and person_id in tracker.prev_frame_ids['persons']:
                person_center = tracker.all_centers['persons'][person_id]

                if tracker.is_unattended(bag_id):
                    frame_visualizer.draw_line(
                        [bag_center[0], person_center[0]],
                        [bag_center[1], person_center[1]],
                        'r'
                    )
                else:
                    frame_visualizer.draw_line(
                        [bag_center[0], person_center[0]],
                        [bag_center[1], person_center[1]],
                        'g'
                    )

        if tracker.is_unattended(bag_id):
            frame_visualizer.draw_text(
                'abandoned',
                tuple(bag_center[0:2]),
                color='r'
            )

    return frame_visualizer.output


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.tracker = SimpleTracker(150, 200)
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions, tracker):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                tracker.update(boxes=predictions.pred_boxes.tensor.numpy(), labels=predictions.pred_classes.numpy())

                if SAVE_PREDICTIONS:
                    SAVED_PREDICTIONS.append(predictions)
                    if len(SAVED_PREDICTIONS) == 100:
                        with open('predictions.pkl', 'wb') as fp:
                            pickle.dump(SAVED_PREDICTIONS, fp)
                            print('Saving done!')

                vis_frame = draw_instance_predictions(video_visualizer, frame, predictions, tracker)

            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions, self.tracker)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions, self.tracker)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame), self.tracker)


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

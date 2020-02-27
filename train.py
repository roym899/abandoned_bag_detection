from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer


register_coco_instances("person_bag_train", {}, "/home/leo/datasets/person_bag/annotations/train.json", "/home/leo/datasets/person_bag/images/train/")
register_coco_instances("person_bag_val", {}, "/home/leo/datasets/person_bag/annotations/val.json", "/home/leo/datasets/person_bag/images/val/")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.DATALOADER.NUM_WORKERS = 2
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("person_bag_train",)
cfg.DATASETS.TEST = ("person_bag_val",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has two classes (person and bag)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # faster, and good enough for this toy dataset (default: 512)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
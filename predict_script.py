import argparse

import cv2
import detectron2.data.transforms as T

# import some common libraries
import torch

# import some common detectron2 utilities
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.modeling.meta_arch.build import build_model
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer

from centermask.config.config import get_cfg
from centermask.modeling.meta_arch.rcnn import (
    TwoStagePseudoLabGeneralizedRCNN,  # hacky imports
)
from centermask.modeling.meta_arch.ts_ensemble import EnsembleTSModel


def filter_insatance(instances, score_threshold):
    filter_mask = instances.scores > score_threshold
    indices = torch.nonzero(filter_mask).flatten().tolist()
    filtered_instances = Instances(
        image_size=instances.image_size,
        pred_classes=instances.pred_classes[indices],
        scores=instances.scores[indices],
        pred_boxes=instances.pred_boxes[indices],
        pred_masks=instances.pred_masks[indices],
    )
    return filtered_instances


class UBTeacherPredictor(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        model = build_model(self.cfg)
        model_teacher = build_model(self.cfg)
        self.ensem_ts_model = EnsembleTSModel(model_teacher, model)

        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.ensem_ts_model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.model = self.ensem_ts_model.modelStudent
        self.model.eval()

        self.aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format


def main(image, model_weights, config_file, score_threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights

    predictor = UBTeacherPredictor(cfg)
    instances = predictor(image)["instances"]
    filtered_instances = filter_insatance(instances, score_threshold)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(filtered_instances.to("cpu"))
    return out.get_image()[:, :, ::-1]


if __name__ == "__main__":

    parser = argparse.ArgumentParser("")
    parser.add_argument("--input_path", help="Input path to image", required=True)
    parser.add_argument("--output_path", help="Output path for prediction", required=True)
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--score_threshold", default=0.5)

    args = parser.parse_args()
    image = cv2.imread(args.input_path)
    output_image = main(image, args.model_weights, args.config_file, args.score_threshold)
    cv2.imwrite(args.output_path, output_image)

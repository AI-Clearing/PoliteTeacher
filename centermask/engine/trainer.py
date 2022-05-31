# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Dominik Filipiak (AI Clearing), 2021. All Rights Reserved.
import logging
import os
import time
from collections import OrderedDict
from typing import Any, List

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase, hooks
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.train_loop import AMPTrainer
from detectron2.evaluation import (
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    verify_results,
)

from detectron2.layers.mask_ops import paste_masks_in_image
from centermask.evaluation import COCOEvaluator
from detectron2.structures.boxes import Boxes
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel
from typing_extensions import final
from centermask.utils.masks_thresholding import binary_mask_to_countour
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_semisup_train_loader_two_crops,
    build_detection_test_loader,
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.solver.build import build_lr_scheduler
from imantics import Polygons, Mask
from clearml import Logger

# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)


# Unbiased Teacher Trainer
class UBTeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.ema_lossess = dict()
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=None, mask_thres=0.5):
        valid_map = proposal_bbox_inst.scores > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]

        pseudo_masks = paste_masks_in_image(proposal_bbox_inst.pred_masks[valid_map][:, 0, :, :], new_proposal_inst.gt_boxes, image_shape, mask_thres)

        all_countours = []
            
        for mask in pseudo_masks:
            all_countours.append(binary_mask_to_countour(mask))

        new_proposal_inst.gt_masks = PolygonMasks(all_countours)
        
        if self.cfg.DEBUG_OPT.PRINTING_MASKS:
            new_proposal_inst.bt_gt_masks = [polygons_to_bitmask(m, image_shape[0], image_shape[1]) for m in new_proposal_inst.gt_masks]


        return new_proposal_inst

    def process_pseudo_label(self, proposals_rpn_unsup_k, cur_threshold, mask_cur_threshold, psedo_label_method=""):
        list_instances = []
        if_empty_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, mask_thres= mask_cur_threshold
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)

            if len(proposal_bbox_inst.gt_classes):
                if_empty_instances.append(False)
            else:
                if_empty_instances.append(True)

        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output, if_empty_instances

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    @staticmethod
    def filter_empty(to_filter: List[Any],  if_empty_instances: List[bool]):
        return [elem for elem, if_empty in zip(to_filter, if_empty_instances) if not if_empty]

    def log_gradients_in_model(self, model):
        for tag, value in model.named_parameters():
            if value.grad is not None:
                self.storage.put_scalar(f"grad/{tag}", value.grad.cpu().sum())

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # remove unlabeled data labels
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict = self.model(label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
            mask_cur_threshold = self.cfg.SEMISUPNET.MASK_THRESHOLD

            joint_proposal_dict = {}

            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, num_proposal_output, if_empty_instances = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, mask_cur_threshold, "thresholding"
            )

            if self.cfg.DEBUG_OPT.FILTER_PSEUDO_INST:
                pesudo_proposals_roih_unsup_k = self.filter_empty(pesudo_proposals_roih_unsup_k, if_empty_instances)
                unlabel_data_q = self.filter_empty(unlabel_data_q, if_empty_instances)
                unlabel_data_k = self.filter_empty(unlabel_data_k, if_empty_instances)

            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            #  add pseudo-label to unlabeled data
            unlabel_data_q = self.add_label(unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"])
            unlabel_data_k = self.add_label(unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"])

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            record_all_label_data = self.model(all_label_data, branch="supervised")
            record_dict.update(record_all_label_data)
            
            if all_unlabel_data: # could be empty bc filtering empty instances
                
                if self.cfg.DEBUG_OPT.PRINTING_MASKS and all_unlabel_data[0]["instances"].bt_gt_masks:
                    Logger.current_logger().report_image("debug", "mask", iteration=self.iter, image= all_unlabel_data[0]["instances"].bt_gt_masks[0]*255)
                    Logger.current_logger().report_image("debug", "image", iteration=self.iter, image= all_unlabel_data[0]["image"].cpu().numpy().transpose((1,2,0)))

                record_all_unlabel_data = self.model(all_unlabel_data, branch="supervised")
                new_record_all_unlabel_data = {}
                for key in record_all_unlabel_data.keys():
                    new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
                record_dict.update(new_record_all_unlabel_data)

            loss_dict = {}
           # POSSIBLE KEYS:
           # ['loss_mask', 'loss_maskiou', 'loss_fcos_cls', 'loss_fcos_loc', 'loss_fcos_ctr', 
           # 'loss_mask_pseudo', 'loss_maskiou_pseudo', 'loss_fcos_cls_pseudo', 'loss_fcos_loc_pseudo', 'loss_fcos_ctr_pseudo'])
            ignored_loss_keys = ['loss_fcos_loc_pseudo', 'loss_fcos_ctr_pseudo']

            if not self.cfg.SEMISUPNET.MASK_LOSS:
                ignored_loss_keys.extend(['loss_mask_pseudo', 'loss_maskiou_pseudo'])

            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key in ignored_loss_keys:
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        human_loss_key = key[:-7]

                        if self.cfg.SEMISUPNET.NORM_LOSS:
                            weight_decay = self.cfg.SEMISUPNET.NORM_LOSS_KEEP_RATE
                            with torch.no_grad():
                                for k in [human_loss_key, key]:
                                # initialize loss during first iteration
                                    self.ema_lossess[k] = self.ema_lossess.get(k, record_dict[k])
                                    self.ema_lossess[k] = weight_decay * self.ema_lossess[k] +  (1 - weight_decay) * record_dict[k]
       
                            loss_dict[key] =  self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT * (self.ema_lossess[human_loss_key] / self.ema_lossess[key])  * record_dict[key]

                        else:
                            loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT

                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            

            losses = sum(loss_dict.values())

            if self.cfg.SEMISUPNET.NORM_LOSS: # should rescale only loss with pseudo part, it cause pick of loss at begining of techer-student phase.
                losses = 1 / (1 + self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT) * losses

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()

        if self.cfg.DEBUG_OPT.LOG_GRADIENT:
            self.log_gradients_in_model(self.model)
        
        if self.cfg.DEBUG_OPT.GRAD_CLIPPING:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v) for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()}

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {key[7:]: value for key, value in self.model.state_dict().items()}
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = student_model_dict[key] * (1 - keep_rate) + value * keep_rate
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {key[7:]: value for key, value in self.model.state_dict().items()}
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k] for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

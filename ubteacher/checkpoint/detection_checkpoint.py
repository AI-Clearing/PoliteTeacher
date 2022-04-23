# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# for load_student_model
from typing import Any, Dict

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts

# for load_student_model
from fvcore.common.checkpoint import _IncompatibleKeys, _strip_prefix_if_present


class DetectionTSCheckpointer(DetectionCheckpointer):
    def _load_model(self, checkpoint):
        if checkpoint.get("__author__", None) == "Caffe2":
            # pretrained model weight: only update student model
            if checkpoint.get("matching_heuristics", False):
                self._convert_ndarray_to_tensor(checkpoint["model"])
                # convert weights by name-matching heuristics
                checkpoint["model"] = align_and_update_state_dicts(
                    self.model.modelStudent.state_dict(),
                    checkpoint["model"],
                    c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
                )

            # for non-caffe2 models, use standard ways to load it
            incompatible = self._load_student_model(checkpoint)

            model_buffers = dict(self.model.modelStudent.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                # Ignore missing key message about pixel_mean/std.
                # Though they may be missing in old checkpoints, they will be correctly
                # initialized from config anyway.
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

        else:  # whole model
            if checkpoint.get("matching_heuristics", False):
                self._convert_ndarray_to_tensor(checkpoint["model"])
                # convert weights by name-matching heuristics
                checkpoint["model"] = align_and_update_state_dicts(
                    self.model.state_dict(),
                    checkpoint["model"],
                    c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
                )
            # for non-caffe2 models, use standard ways to load it
            incompatible = super()._load_model(checkpoint)

            model_buffers = dict(self.model.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                # Ignore missing key message about pixel_mean/std.
                # Though they may be missing in old checkpoints, they will be correctly
                # initialized from config anyway.
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

    def _load_student_model(self, checkpoint: Any) -> _IncompatibleKeys:  # pyre-ignore
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # work around https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.modelStudent.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        # pyre-ignore
        incompatible = self.model.modelStudent.load_state_dict(
            checkpoint_state_dict, strict=False
        )
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )

    def _load_file(self, f: str) -> Dict[str, Any]:
        checkpoint = super()._load_file(f)
        checkpoint = checkpoint.pop("model")

        overwrite_model_name = "modelStudent"
        possible_model_name = [overwrite_model_name, "modelTeacher"]

        def get_new_weigths_key(key):
            if not any([key.startswith(name) for name in possible_model_name]):
                return f"{overwrite_model_name}.{key}"
            else:
                return key

        modify_checkpoint = {
            get_new_weigths_key(k): v for k, v in checkpoint.items() 
        }
        checkpoint = {"model": modify_checkpoint}
        return checkpoint

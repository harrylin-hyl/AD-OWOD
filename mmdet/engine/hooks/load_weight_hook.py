# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class LoadWeightHook(Hook):
    """Load Weight Hook."""
    
    def before_train(self, runner: Runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        # the initial weights are the same as encoder of pretrained model
        for (src_name, src_parm), (pro_name, pro_parm) in zip(
            model.encoder.named_parameters(),
            model.prompt_encoder.named_parameters()):
            pro_parm.data = src_parm.data.clone()
        # the initial weights are the same as decoder of pretrained model
        for (src_name, src_parm), (pro_name, pro_parm) in zip(
            model.decoder.named_parameters(),
            model.prompt_decoder.named_parameters()):
            pro_parm.data = src_parm.data.clone()
        # weights of memory_trans_fc
        for (src_name, src_parm), (pro_name, pro_parm) in zip(
            model.memory_trans_fc.named_parameters(),
            model.prompt_memory_trans_fc.named_parameters()):
            pro_parm.data = src_parm.data.clone()
        # weights of head
        for (src_name, src_parm), (pro_name, pro_parm) in zip(
            model.bbox_head.cls_branches.named_parameters(),
            model.bbox_head.prompt_cls_branches.named_parameters()):
            pro_parm.data = src_parm.data.clone()
        for (src_name, src_parm), (pro_name, pro_parm) in zip(
            model.bbox_head.reg_branches.named_parameters(),
            model.bbox_head.prompt_reg_branches.named_parameters()):
            pro_parm.data = src_parm.data.clone()
        # weights of query
        for (src_name, src_parm), (pro_name, pro_parm) in zip(
            model.query_embedding.named_parameters(),
            model.prompt_query_embedding.named_parameters()):
            pro_parm.data = src_parm.data.clone()
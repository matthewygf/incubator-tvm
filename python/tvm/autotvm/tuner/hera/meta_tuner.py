"""Tuner that can have any optimizer with a sampler"""

import logging
from tvm.autotvm.tuner.model_based_tuner import CostModel
from tvm.autotvm.tuner.model_based_tuner import ModelBasedTuner, ModelOptimizer
from tvm.autotvm.tuner.sampler import Sampler
from .neural_sampler import NeuralSampler
from ..chameleon.adaptive_sampler import AdaptiveSampler

class MetaTuner(ModelBasedTuner):
    def __init__(self, task, cost_model, optimizer, platform, plan_size=64,
                 sampler="neural",
                 diversity_filter_ratio=None) -> None:

        assert isinstance(optimizer, ModelOptimizer), "Optimizer must be " \
                                                      "a ModelOptimizer object."
        
        assert isinstance(cost_model, CostModel), "cost_model must be " \
                                                  "a CostModel object."
        
        if sampler == None:
            sampler = None
        elif sampler == 'neural':
            import os
            current_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_path)
            model_path = os.path.join(current_dir, "newmodel.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(current_dir, "default.pt")
            embed_path = os.path.join(current_dir, "new_train.csv")
            if not os.path.exists(embed_path):
                embed_path = os.path.join(current_dir, "default.csv")
                
            logging.info(f"Path of model: {model_path}, Path of embedding: {embed_path}")
            sampler = NeuralSampler(model_path, embed_path, task, platform)
        elif sampler == 'adaptive':
            sampler = AdaptiveSampler(plan_size)
        else:
            assert isinstance(sampler, Sampler), "Sampler must be None" \
                                                 "a supported name string," \
                                                 "or a Sampler object."

        super(MetaTuner, self).__init__(task, cost_model, optimizer, plan_size,
                                        diversity_filter_ratio, sampler=sampler)
    
    def tune(self, *args, **kwargs):
        super(MetaTuner, self).tune(*args, **kwargs)

        # manually close pool to avoid multiprocessing issues
        self.cost_model._close_pool()
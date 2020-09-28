"""Tuner that can have any optimizer with a sampler"""

from python.tvm.autotvm.tuner.model_based_tuner import CostModel
from tvm.autotvm.tuner.model_based_tuner import ModelBasedTuner, ModelOptimizer
from tvm.autotvm.tuner.sampler import Sampler


class MetaTuner(ModelBasedTuner):
    def __init__(self, task, cost_model, optimizer, plan_size=64,
                 sampler="neural",
                 diversity_filter_ratio=None) -> None:

        assert isinstance(optimizer, ModelOptimizer), "Optimizer must be " \
                                                      "a ModelOptimizer object."
        
        assert isinstance(cost_model, CostModel), "cost_model must be " \
                                                  "a CostModel object."
        
        if sampler == None:
            sampler = None
        elif sampler == 'neural':
            sampler = NeuralSampler()
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
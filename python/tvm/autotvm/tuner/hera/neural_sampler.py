# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return,invalid-name,consider-using-enumerate,abstract-method
"""Sampler that uses Neural Network to obtain suitable samples"""

import numpy as np
from ..sampler import Sampler
from ...env import GLOBAL_SCOPE

import torch
from .model import *
import pandas as pd 

class NeuralSampler(Sampler):
    """Sampler that uses Neural Network for filtering
    """

    def __init__(self, model_path, embedding_path) -> None:
        self.model_path = model_path
        self.model = torch.load(model_path)
        self.df = pd.read_csv(embedding_path)

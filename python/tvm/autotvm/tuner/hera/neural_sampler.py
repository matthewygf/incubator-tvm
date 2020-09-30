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

from logging import error

from numpy.lib.utils import who
from tvm.autotvm.task.space import OtherOptionEntity, SplitEntity
from tvm.autotvm.task.space import OtherOptionSpace, SplitSpace
import numpy as np
from ..sampler import Sampler
from ...env import GLOBAL_SCOPE

import torch
from .model import *
import pandas as pd 
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger("autotvm")

class NeuralSampler(Sampler):
    """Sampler that uses Neural Network for filtering
    """

    def __init__(self, 
                 model_path, 
                 embedding_path, # NOTE: Ideally this should be some dfs somewhere
                 task, 
                 platform, 
                 allow_train=True,
                 train_epoch=1) -> None:
        self.model_path = model_path
        self.model = torch.load(model_path)
        self.df = pd.read_csv(embedding_path)
        self.old_feature_space = list(self.df.columns)
        self.platforms = list(self.df.platform.unique())
        self.num_platform = len(self.platforms)
        self.platforms_to_int = {}
        self.df.platform = self.df.platform.apply(self.platform_to_int)
        self.hashed_candidates = {}
        self.operators_candidates = list(self.df.cand_id.unique())
        self.task = task
        self.trainable = True
        self.preprocess_candidates()
        
        self.representations = []
        for f in self.old_feature_space:
            if f.startswith("feature_"):
                self.representations.append(f)
        logger.info(f"We will use feature space{self.representations}")

        if self.platforms_to_int.get(platform, None) is None:
            self.platforms_to_int[platform] = self.num_platform
            self.num_platform+=1
            logger.info(f"Number of platforms: {self.platforms_to_int}")
        self.task_config_space = task.config_space
        self.space_map_keys = list(self.task_config_space.space_map.keys())
        feature_mappings = [True if k in self.old_feature_space else False for k in self.space_map_keys]
        self.not_in_network = [ v for include, v in zip(feature_mappings, self.space_map_keys) if not include ]
        self.require_train = len(self.not_in_network) > 0
        self.allow_train = allow_train
        self.train_epoch = train_epoch
        logger.info(f"Features not in network: {self.not_in_network}")
        
        if self.require_train:
            # add the feature cols into the df.
            self.to_add_cols = {}
            for n in self.not_in_network:
                space = self.task_config_space.space_map[n]
                if isinstance(space, SplitSpace):
                    self.to_add_cols[n] = space.num_output+1
                    for i in range(1, space.num_output+1):
                        # 9999 will be padding idx used in the network
                        self.df[n+"_"+str(i)] = 9999
                elif isinstance(space, OtherOptionSpace):
                    # 2 will be the padding idx
                    # NOTE: assuming otheroptionspace only have 0,1
                    self.df[n] = len(space)
                    self.to_add_cols[n] = len(space)
                else:
                    raise NotImplementedError()
                    
            logger.info(f"{self.df.head()}")

    def preprocess_candidates(self):
        m = 0
        for i in self.operators_candidates:
            temp = self.df[self.df.cand_id == i]
            unique_cand_name = list(temp.cand_name.unique())
            unique_arguments = list(temp.arguments.unique())
            assert len(unique_cand_name) == 1
            assert len(unique_arguments) == 1
            hashed_str = hash(unique_cand_name[0]+unique_arguments[0])
            self.hashed_candidates[hashed_str] = i
            m = max(m, i)

        self.hashed_task = hash(self.task.name + str(self.task.args))
        if self.hashed_candidates.get(self.hashed_task, None) is None:
            m+=1
            self.hashed_candidates[self.hashed_task] = m
            self.operators_candidates.append(m)
            self.cand_id = m
        else:
            self.cand_id = self.hashed_candidates.get(self.hashed_task)
        logger.info(f"Hashed candidates to cand_id: {self.hashed_candidates}")

    def platform_to_int(self, x):
        if x == "intel_1080":
            self.platforms_to_int[x] = 0
            return 0
        elif x == "amd_2080":
            self.platforms_to_int[x] = 1
            return 1
        elif x == "intel_2080":
            self.platforms_to_int[x] = 2
            return 2
        elif x == "intel_v100":
            self.platforms_to_int[x] = 3
            return 3
        else:
            cat = self.platforms_to_int.get(x, None)
            if cat is None:
                self.platforms_to_int[x] = self.num_platform
                self.num_platform+=1
                return self.num_platform

            self.platforms_to_int[x] = cat
            return cat

    def sample(self, samples, dims):
        """Samples using Neural Network"""
        logger.info(f"Sampling.... from {samples}, dimensions: {dims}")
        #TODO:
        return samples


    def fit(self, xs, ys):
        if self.require_train and self.allow_train:
            logger.info(f"training the network first for {self.train_epoch} by appending the samples to the dataframes")
            self.preprocess(xs, ys)

            # prepare for training
            target = self.df["candidate_time_cost_avg"]
            self.df.drop(["candidate_time_cost_avg"], axis=1, inplace=True)
            non_train=["arguments", "cand_name", "error", "index"]
            cat_features = []
            for c in list(self.df.columns):
                if c.startswith("feature"):
                    continue

                if c not in non_train and cat_features != "flop":
                    cat_features.append(c)
            
            logger.info(f"Category features {cat_features}")
            self.df[cat_features] = self.df[cat_features].astype(int)


            # grep the size of the category and padding index
            cats_size = {}
            padding_indexes = {}
            for c in cat_features:
                m = max(self.df[c])
                if m == -1:
                    continue
                
                uniques= self.df[c].unique()
                unique_length = len(uniques)
                if unique_length == 2 and only_nan_and_whatever(uniques):
                    self.df[c] = self.df[c].apply(lambda x: 0 if x == -1 else x)
                cats_size[c] = m+2
                padding_indexes[c] = m+1

            logger.info(f"CategorySize: {cats_size} \n Padding Indexes: {padding_indexes}")

            self.model = self.train(self.df[cat_features], target, cats_size, padding_indexes, epoch=self.train_epoch)
            self.update_embeddings(cat_features, target)
            

    def preprocess(self, xs, results):
        rows = []
        for i, s in enumerate(xs):
            task_info = [
                np.math.log1p(self.task.flop),
                self.cand_id,
                self.num_platform,
                self.task.name,
                str(self.task.args),
                np.mean(results[i].costs) * self.task.flop if results[i].error_no != 0 else -1, # mean costs
                transform_error(results[i].error_no),
                0, #outlier default
            ]

            # Minus 7 task info above.
            old_features = [9999 for _ in range(len(self.old_feature_space)-8)]
            new_features = []
            config = self.task_config_space.get(s)
            for k,v in config._entity_map.items():
                if isinstance(v, SplitEntity):
                    for i in range(1, self.to_add_cols[k]):
                        new_features.append(v.size[i-1])
                elif isinstance(v, OtherOptionEntity):
                    new_features.append(int(v.val))
            
            all_feats = task_info+old_features+new_features
            assert len(all_feats) == len(self.df.columns)
            rows.append(all_feats)
        dfs = pd.DataFrame(rows, columns=list(self.df.columns))
        def interquantile_outlier(df):
            non_errors = df[df["error"] == "none"] 
            q1 = non_errors["candidate_time_cost_avg"].quantile(0.25)
            q3 = non_errors["candidate_time_cost_avg"].quantile(0.75)
            iqr = q3-q1
            upperb = q3 + (1.5*iqr)
            outliers = non_errors[non_errors["candidate_time_cost_avg"] > upperb]
            outliers["outliers"] = 1
            df.loc[outliers.index, "outliers"] = outliers["outliers"]
            return df

        # set up outliers
        # logging.info(f"{dfs.head()}")
        dfs = dfs.groupby(["cand_id", "platform"]).apply(interquantile_outlier).reset_index()
        # outunique = dfs["outliers"].unique()
        # logging.info(f"{outunique}")
        self.df = pd.concat([dfs, self.df], ignore_index=True)
        logger.info(self.df.columns)
        # logger.info(f"{self.df.tail()}")


    def train(self, xs, ys, category_size, padding_idxes, epoch=30):
        x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, shuffle=True, random_state=2022)
        train_dataset = WhateverDS(x_train, y_train)
        val_dataset = WhateverDS(x_test, y_test)
        batch_size = 512
        lr_rate = 1e-2
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        model = WhateverModel(category_size,batch_size, padding_idxes, embedding_dims=10)
        opt = torch.optim.Adam(model.parameters(), lr=lr_rate)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1, verbose=True, min_lr=1e-9)
        loss = torch.nn.MSELoss()
        best_loss = 0.0
        log_iter = 10
        model.train()
        for e in range(1,epoch+1):
            running_loss = 0.0
            all_loss = 0.0
            for i, (x, y) in enumerate(train_loader):
                opt.zero_grad()
                pred = model(x)
                y = y.reshape(y.size(0), 1)
                loss_v = loss(pred, y.float())
                loss_v.backward()
                opt.step()
                running_loss += loss_v.item()
                all_loss += loss_v
                        
                if i % log_iter == 0:
                    running_best = running_loss / log_iter
                    logger.info(f"epoch {e} iter {i+1} loss {running_best}")
                    running_loss = 0.0
                    
            val_losses = 0.0
            avg_loss = 0.0
            for i, (x2, y2) in enumerate(val_loader):
                pred=model(x2)
                y2 = y2.reshape(y2.size(0), 1)
                val_loss = loss(pred, y2)
                val_losses+=val_loss.item()
                
                avg_loss = val_losses/len(val_loader)

            logger.info(f"avg e {e+1}, val loss: {avg_loss}")
            sched.step(avg_loss)
            if avg_loss < best_loss and e > 5:
                torch.save(model.state_dict(), f"newmodel_{e}.pt")
                best_loss = avg_loss

        del opt, sched, loss, train_loader, val_loader, val_dataset, train_dataset
        model.eval()
        return model


    def update_embeddings(self, category_features, target, batch_size=512):
        logger.info(f"Updating Embeddings ... {self.representations}")
        who_dataset = WhateverDS(self.df[category_features], target)
        dloader = DataLoader(who_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        index = 0
        for i, (x,_) in enumerate(dloader):
            embed_feats = self.model(x, True)
            batch = embed_feats.shape[0]
            # NOTE: fail here ?!            
            self.df.loc[index:index+batch-1, self.representations] = embed_feats.detach().numpy().astype(np.float32)
            index += batch
        logger.info("updated Embeddings ... ")
        logger.info(f"{self.df.tail()}")
        del who_dataset, dloader
        self.df.to_csv("train.csv", index=False)

def transform_error(error_no):
    if error_no == 0:
        return "none"
    elif error_no == 2:
        return "compile_timeout"
    elif error_no == 3:
        return "compile_timeout"
    elif error_no == 4:
        return "runtime_error"
    elif error_no == 7:
        return "run_timeout"
    elif error_no == 5:
        return "wrong_answer"
    elif error_no == 6:
        return "build_timeout"
    else:
        raise NotImplementedError(error_no)

def only_nan_and_whatever(uniques):
    if -1 not in uniques:
        return False
    
    if 9999 not in uniques:
        return False
    
    return True
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
from .similarity import similar_to
import os

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
                 train_epoch=2) -> None:
        self.model_path = model_path
        self.df = pd.read_csv(embedding_path)
        self.old_feature_space = list(self.df.columns)
        self.column_order = self.old_feature_space
        self.platforms = list(self.df.platform.unique())
        self.num_platform = len(self.platforms)
        self.hashed_candidates = {}
        self.operators_candidates = list(self.df.cand_id.unique())
        self.task = task
        self.trainable = True
        self.preprocess_candidates()
        self.eps = 0.99
        self.train_cnt = 0
        self.n_train = 5 # every n times the cost model has been updated
        current_path = os.path.abspath(__file__)
        self.current_dir = os.path.dirname(current_path)
        self.model = None
        
        self.representations = []
        for f in self.old_feature_space:
            if f.startswith("feature_"):
                self.representations.append(f)
        logger.info(f"We will use feature space{self.representations}")

        if platform not in self.platforms:
            self.platform_id = max(self.platforms) + 1
            self.platforms.append(platform)
            self.num_platform+=1
        else:
            self.platform_id = platform_to_int(platform)
        logger.info(f"Current platform {self.platform_id} Number of platforms: {self.num_platform} --- {self.platforms}")

        self.df["platform"] = self.df["platform"].apply(platform_to_int)
        self.task_config_space = task.config_space
        self.space_map_keys = list(self.task_config_space.space_map.keys())
        self.not_in_network = []
        for k in self.space_map_keys:
            found = False
            for of in self.old_feature_space:
                if of.startswith(k):
                    found = True
                    break
            if not found:
                self.not_in_network.append(k)
                
        self.allow_train = allow_train
        self.train_epoch = train_epoch
        logger.info(f"Features not in network: {self.not_in_network}")
        
        self.to_add_cols = {}
        for k, space in self.task_config_space.space_map.items():
            if isinstance(space, SplitSpace):
                self.to_add_cols[k] = space.num_output+1
                for i in range(1, space.num_output+1):
                    # 9999 will be padding idx used in the network
                    self.df[k+"_"+str(i)] = 9999
            elif isinstance(space, OtherOptionSpace):
                # 2 will be the padding idx
                # NOTE: assuming otheroptionspace only have 0,1
                self.df[k] = len(space)
                self.to_add_cols[k] = len(space)
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

    def sample(self, samples, dims):
        """Samples using Neural Network"""
        # 1. reconstruct the input 
        res = []
        rows = []
        if len(samples) == 0:
            return res

        if not self.model and self.model_path:
            # prepare category features and padding indexes
            non_train=["arguments", "cand_name", "error", "index", "outliers", "candidate_time_cost_avg", "flop"]
            cat_features, cats_size, padding_indexes = self.prepare_cat_features(non_train)
            self.model = WhateverModel(cats_size, padding_indexes, embedding_dims=10)
            self.model.load_state_dict(torch.load(self.model_path))

        for s in samples:
            task_info = [
                np.math.log1p(self.task.flop),
                self.cand_id,
                self.platform_id,
                self.task.name,
                str(self.task.args),
                -1, 
                transform_error(0),
                0, #outlier default
            ]

            new_features = []
            config = self.task_config_space.get(s[1])
            for k,v in config._entity_map.items():
                if isinstance(v, SplitEntity):
                    for i in range(1, self.to_add_cols[k]):
                        new_features.append(v.size[i-1])
                elif isinstance(v, OtherOptionEntity):
                    new_features.append(int(v.val))

            remaining = len(self.df.columns) - len(task_info) - len(new_features)

            old_features = [9999 for _ in range(remaining)]
            all_feats = task_info+old_features+new_features
            assert len(all_feats) == len(self.df.columns), f'Got {len(all_feats)} , \
                                                             expected len(self.df.columns)'
            rows.append(all_feats)
            res.append(s[0])

        dfs = pd.DataFrame(rows, columns=self.column_order)
        non_used=["arguments", "cand_name", "error", "index", "outliers", "candidate_time_cost_avg", "flop"]
        cat_features = self.obtain_cat_features(non_used)

        # replace -1 indexes to 0 for category features
        for c in cat_features:
            dfs[c] = dfs[c].apply(lambda x: 0 if x == -1 else x)

        logger.info(f"Samples DF: \n {dfs.head()}")
        self.model.eval()
        
        samples_dataset = WhateverPredDs(dfs)
        samples_loader = DataLoader(samples_dataset, batch_size=1, shuffle=False, drop_last=False)
        remove_index=[]
        k=5
        for i, sample_tensor in enumerate(samples_loader):
            p = np.random.random()
            if p < self.eps:
                continue
            
            embeddings = self.model(sample_tensor, True)
            similar_samples = similar_to(self.df, self.cand_id, self.platform_id, self.representations, embeddings.detach().numpy(), k=k)
            # logger.info(f"sim samples: {similar_samples}")

            # slightly over half
            if len(similar_samples[similar_samples["outliers"] == 1]) >= k // 2 + 1:
                remove_index.append(i)
        
        del samples_loader, samples_dataset

        # remove from the full samples
        if len(remove_index) > 0:
            logger.info(f"removing : {remove_index}")
            a = []
            for i, configs in enumerate(res):
                if i in remove_index:
                    continue
                a.append(configs)
            logger.info(f"remain: {a}")
            return a
        return res

    def obtain_cat_features(self, non_train_cols):
        cat_features = []
        for c in list(self.df.columns):
            if c.startswith("feature"):
                continue

            if c not in non_train_cols:
                cat_features.append(c)
        return cat_features

    def prepare_cat_features(self, non_train_cols):
        cat_features = self.obtain_cat_features(non_train_cols)
        
        logger.info(f"Category features {cat_features}")
        self.df[cat_features] = self.df[cat_features].astype(int)
        cats_size = {}
        padding_indexes = {}
        for c in cat_features:
            m = max(self.df[c])
            self.df[c] = self.df[c].apply(lambda x: 0 if x == -1 else x)
            cats_size[c] = m+2
            padding_indexes[c] = m+1
        return cat_features, cats_size, padding_indexes

    def fit(self, xs, ys):
        self.eps = max(0.005, self.eps - 1)
        self.train_cnt += 1

        if self.allow_train and self.train_cnt % self.n_train == 0:
            logger.info(f"training the network first for {self.train_epoch} by appending the samples to the dataframes")
            self.preprocess(xs, ys)

            # prepare category features and padding indexes
            non_train=["arguments", "cand_name", "error", "index", "outliers", "candidate_time_cost_avg", "flop"]
            cat_features, cats_size, padding_indexes = self.prepare_cat_features(non_train)

            logger.info(f"CategorySize: {cats_size} \n Padding Indexes: {padding_indexes}")
            trains = self.df[cat_features+["flop"]]
            logger.info(f"{trains.flop.describe()}")
            self.model = self.train(trains, self.df["candidate_time_cost_avg"], cats_size, padding_indexes, epoch=self.train_epoch)
            self.update_embeddings(trains, self.df["candidate_time_cost_avg"])
        else:
            logger.info(f"train cnt: {self.train_cnt}")
        
            
    def preprocess(self, xs, results):
        rows = []
        for i, s in enumerate(xs):
            task_info = [
                np.math.log1p(self.task.flop),
                self.cand_id,
                self.platform_id,
                self.task.name,
                str(self.task.args),
                np.mean(results[i].costs) if results[i].error_no == 0 else -1, # mean costs
                transform_error(results[i].error_no),
                0, #outlier default
            ]

            new_features = []
            config = self.task_config_space.get(s)
            for k,v in config._entity_map.items():
                if isinstance(v, SplitEntity):
                    for i in range(1, self.to_add_cols[k]):
                        new_features.append(v.size[i-1])
                elif isinstance(v, OtherOptionEntity):
                    new_features.append(int(v.val))

            # Minus task info above.
            old_features = [9999 for _ in range(len(self.column_order)-len(task_info) -len(new_features))]
            
            all_feats = task_info+old_features+new_features
            assert len(all_feats) == len(self.df.columns)
            rows.append(all_feats)
        self.column_order = list(self.df.columns)
        dfs = pd.DataFrame(rows, columns=self.column_order)
        logger.info(f"new rows: \n {dfs.tail()}")

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
        dfs = dfs.groupby(["cand_id", "platform"]).apply(interquantile_outlier).reset_index()
        self.df = pd.concat([dfs, self.df], ignore_index=True)
        # NOTE: concat reshuffle the cols :/
        self.df = self.df[self.column_order]
        logger.info(self.df["candidate_time_cost_avg"].describe())
        logger.info(f"finished prep: \n {self.df.tail()}")

    def train(self, xs, ys, category_size, padding_idxes, epoch=30):
        # every time we train, we reduce the eps
        x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, shuffle=True, random_state=2022)
        train_dataset = WhateverDS(x_train, y_train)
        val_dataset = WhateverDS(x_test, y_test)
        batch_size = 512
        lr_rate = 1e-2
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        model = WhateverModel(category_size, padding_idxes, embedding_dims=10)
        opt = torch.optim.Adam(model.parameters(), lr=lr_rate)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1, verbose=True, min_lr=1e-9)
        loss = torch.nn.MSELoss()
        best_loss = 9999
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
            if avg_loss < best_loss:
                model_path = os.path.join(self.current_dir, "newmodel.pt")
                self.model_path = model_path
                torch.save(model.state_dict(), self.model_path)
                best_loss = avg_loss

        del opt, sched, loss, train_loader, val_loader, val_dataset, train_dataset
        model.eval()
        return model

    def update_embeddings(self, trains, target, batch_size=512):
        logger.info(f"Updating Embeddings ... {self.representations}")
        who_dataset = WhateverDS(trains, target)
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
        # re-convert the platform to string
        self.df["platform"] = self.df["platform"].apply(int_to_platform)
        embed_path = os.path.join(self.current_dir, "new_train.csv")
        self.df.to_csv(embed_path, index=False)
        # quickly re-convert back so we can continue :/
        self.df["platform"] = self.df["platform"].apply(platform_to_int)

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

def int_to_platform(x):
    if x == 0:
        return "intel_1080"
    elif x == 1:
        return "amd_2080"
    elif x == 2:
        return "intel_2080"
    elif x == 3:
        return "intel_v100"
    elif x == 4:
        return "i7laptop"
    else:
        raise NotImplementedError()

def platform_to_int(x):
    if x == "intel_1080":
        return 0
    elif x == "amd_2080":
        return 1
    elif x == "intel_2080":
        return 2
    elif x == "intel_v100":
        return 3
    elif x == "i7laptop":
        return 4
    else:
        raise NotImplementedError()


def only_nan_and_whatever(uniques):
    if -1 not in uniques:
        return False
    
    if 9999 not in uniques:
        return False
    
    return True
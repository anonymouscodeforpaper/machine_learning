#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
from torch.nn import LeakyReLU
leaky = LeakyReLU(0.2)
import torch.nn as nn
import torch

import pandas as pd
import torch
from torch import nn

import torch.nn.functional as F

rate = 0


# In[ ]:



def calculate_score(x, users, aspects):
    niubi = []
    first = x.index[0]
    val_base = x[first]
    actor_base = torch.LongTensor(val_base)
    actors_base = aspects(actor_base)
    pre_rating = torch.mm(actors_base, users[first].unsqueeze(1))
    niubi.append(pre_rating)
    pre_rating = pre_rating / pre_rating.shape[0]
    pre_rating = pre_rating.sum(0)
    for i in x.index[1:]:
        val = x[i]
        actor = torch.LongTensor(val)
        actors = aspects(actor)
        pre_ra = torch.mm(actors, users[i].unsqueeze(1))
        niubi.append(pre_ra)
        actors_f = pre_ra / pre_ra.shape[0]
        actors_f = actors_f.sum(0)
        pre_rating = torch.cat((pre_rating, actors_f))
    return pre_rating, niubi


class aspect_augumentation(nn.Module):
    def __init__(self, n_users, n_entity, n_rk, n_factors):
        super(aspect_augumentation, self).__init__()
        self.n_users = n_users
        self.n_entity = n_entity
        self.n_rk = n_rk
        self.n_factors = n_factors
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.entity_factors = torch.nn.Embedding(n_entity, n_factors)
        self.relation_k = torch.nn.Embedding(n_factors, n_rk)

    def forward(self, user_id, artists_id, categories_id):
        '''
        user_factors: n_users * 64
        entity_factors: n_entity * 64
        relation_k: 64 * 3
        '''

        users = self.user_factors(user_id)  # 128 * 8
        aspects = self.entity_factors  # n_entity * 8
        users = F.dropout(users, p=rate, training=self.training)
        scores = torch.matmul(users, F.dropout(
            self.relation_k.weight, p=rate, training=self.training))  # 128 * 3
        scores = leaky(scores)
        m = torch.nn.Softmax(dim=1)  # 128 * 3
        scores = m(scores)  # 128 * 3

        '''
        Compute the importance of each aspects
        '''
        scores_actors = scores[:, 0]  # 128,
        scores_directors = scores[:, 1]  # 128,

        '''
        Compute the contribution of each aspects
        '''
        contribute_actors, niubi_act = calculate_score(
            artists_id, users, aspects)
        contribute_directors, niubi_dir = calculate_score(
            categories_id, users, aspects)

        '''
        Compute the final predictions
        '''
        importance_sum = scores_actors + scores_directors
        prediction_sum = contribute_actors * scores_actors +             contribute_directors * scores_directors
        prediction = prediction_sum / importance_sum
        cnm = [niubi_act, niubi_dir]

        return prediction, scores, contribute_actors, contribute_directors, cnm


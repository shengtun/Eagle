
import torch
import torch.nn as nn
from math import sqrt

from torch.nn import functional as F
import numpy as np


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_q_k, dim_v):

        self.q = nn.Linear(input_dim, dim_q_k)
        self.k = nn.Linear(input_dim, dim_q_k)
        self.v = nn.Linear(input_dim, dim_v)
        self.norm = sqrt(dim_q_k)
        self.embedding_list = []

    def attention_step(self, batch, batch_idx):  # save locally aware patch features
        x, _, file_name, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            # print(feature.shape)
            avep = torch.nn.AvgPool2d(3, 1, 1)
            maxp = torch.nn.MaxPool2d(3, 1, 1)

            if feature.shape[3] == 28:
                saconv_out = avep(feature)

            elif feature.shape[3] == 14:
                attention_in1 = maxp(feature[:])
                attention_in2 = maxp(feature[:])
                attention_v = maxp(feature[:])
                width = attention_in1.shape[3]

                # Flatten: [B, C, H, W] → [B, C, H*W]
                attention_in1_flatten = torch.flatten(attention_in1[:], 2, 3)
                attention_in2_flatten = torch.flatten(attention_in1[:], 2, 3)
                attention_v = torch.flatten(attention_v[:], 2, 3)

                # Reshape for attention: [B, C, HW, 1] and [B, C, 1, HW]
                attention_in1 = torch.reshape(attention_in1_flatten, (
                attention_in1_flatten.shape[0], attention_in1_flatten.shape[1], attention_in1_flatten.shape[2], 1))
                attention_in2 = torch.reshape(attention_in2_flatten, (
                attention_in2_flatten.shape[0], attention_in2_flatten.shape[1], 1, attention_in2_flatten.shape[2]))
                attention_v = torch.reshape(attention_v,
                                            (attention_v.shape[0], attention_v.shape[1], attention_v.shape[2], 1))

                attention_out = torch.matmul(attention_in1, attention_in2)
                attention_out = torch.nn.functional.softmax(attention_out, dim=-1)
                attention_out = torch.matmul(attention_out, attention_v)
                saconv_out = torch.reshape(attention_out,
                                           (attention_out.shape[0], attention_out.shape[1], width, width))

            embeddings.append(saconv_out)
        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))

    def forward(self, x):
        # (b, s, input_dim) -> (b, s, dim_q_k)
        q = self.q(x)
        k = self.k(x)
        # (b, s, input_dim) -> (b, s, dim_v)
        v = self.v(x)

        # (b, s, s)
        d = torch.bmm(q, k.transpose(1, 2)) / self.norm
        softmax_out = torch.softmax(d, dim=-1)

        # (b, s, dim_v)
        attn_out = torch.bmm(softmax_out, v)
        return attn_out

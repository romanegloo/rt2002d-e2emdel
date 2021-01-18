import code

import torch
import torch.nn as nn
import GoshiBoshi.utils as utils

class CRF(nn.Module):
    """Conditional Random Field (CRF) layer
    """

    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transition = nn.Parameter(torch.Tensor(self.tagset_size,
                                                     self.tagset_size))
        utils.init_linear(self.hidden2tag)
        self.transition.data.zero_()

    def forward(self, feats, target, mask):
        """
        args:
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            loss, output
            # output from crf layer ( (batch_size * seq_len), tag_size, tag_size)
        """
        emission_scores = self.hidden2tag(feats)
        batch_size, seq_len = target.shape[:2]
        tag_ids = target.long().argmax(dim=-1)
        scores = torch.gather(emission_scores, dim=2,
                              index=tag_ids.unsqueeze(dim=-1)).squeeze(dim=2)
        # add transition scores
        scores[:, 1:] += self.transition[tag_ids[:, :-1], tag_ids[:, 1:]]
        total_score = (scores * mask.float()).sum(dim=1)  # shape: (b,)

        # calculate the partition
        p = torch.unsqueeze(emission_scores[:, 0], dim=1)
        for i in range(1, seq_len):
            n_uf = mask[:, i].sum()
            p_uf = p[:n_uf]
            emit_and_trans = emission_scores[:n_uf, i].unsqueeze(dim=1) +\
                             self.transition
            log_sum = p_uf.transpose(1, 2) + emit_and_trans  # shape: (uf, K, K)
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)
            log_sum = log_sum - max_v  # shape: (uf, K, K)
            p_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)  # shape: (uf, 1, K)
            p = torch.cat((p_uf, p[n_uf:]), dim=0)
        p = p.squeeze(dim=1)  # shape: (b, K)
        max_p = p.max(dim=-1)[0]
        p = max_p + torch.logsumexp(p - max_p.unsqueeze(dim=1), dim=1) # shape: (b,)
        llk = total_score - p  # shape: (b,)
        # average
        loss = -llk.mean()  # shape: (b,)

        return loss, emission_scores

    def decode(self, scores, mask):
        """Find the optimal path with viterbe decode

        args:
            scores (batch_size, seq_len, target_size)
        """
        batch_size, seq_len, tgt_size = scores.shape
        tags = [[[i] for i in range(tgt_size)]] * batch_size  # list, shape: (b, K, 1)
        d = torch.unsqueeze(scores[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, mask.sum(dim=1)[0]):
            n_uf = mask[:, i].sum()
            d_uf = d[:n_uf]  # shape: (uf, 1, K)
            emit_and_transition = self.transition + scores[:n_uf, i].unsqueeze(dim=1)  # shape: (uf, K, K)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()  # list, shape: (nf, K)
            tags[:n_uf] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_uf)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_uf:]), dim=0)  # shape: (b, 1, K)
        d = d.squeeze(dim=1)  # shape: (b, K)
        _, max_idx = torch.max(d, dim=1)  # shape: (b,)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

import torch
from torch import nn
from torch.nn import functional as F

def mine_full_batch(x: torch.Tensor):
    index = torch.arange(x.size(0)).to(x.device)
    index = torch.cartesian_prod(index, index)

    negative_mask = (index[:, 0] != index[:, 1]).unsqueeze(-1)
    negative_index = index.masked_select(negative_mask).view(1, -1, 2)

    negatives = negative_index[..., 1].flatten()
    return x.index_select(0, negatives).view(x.size(0), -1, *x.size()[1:])

def full_batch_negative_sampler(func):
    def wrapper(query, document, label):
        document_neg = mine_full_batch(document)
        
        full_documents = torch.cat([document.unsqueeze(1), document_neg], dim=1)
        full_queries = query.unsqueeze(1).expand_as(full_documents)

        full_queries, full_documents = map(lambda x: x.flatten(0, -2), (full_queries, full_documents))

        label_neg = torch.zeros(document_neg.size()[:-1], device=label.device)
        full_labels = torch.cat([label.unsqueeze(1), label_neg], dim=1).flatten()

        return func(full_queries, full_documents, full_labels)
    return wrapper
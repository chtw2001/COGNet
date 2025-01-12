import torch
import numpy as np
import torch.nn.functional as F

batch_size, max_visit_num, max_diag_num, emb_size = 100, 10, 5, 1
# tensor = torch.rand([batch_size * max_visit_num, max_diag_num, emb_size])
# emb_size = 64
# tensor3 = torch.rand([batch_size * max_visit_num, max_diag_num, emb_size])
# squeezed_tensor = tensor.squeeze(-1)

# print(squeezed_tensor.shape)

# tensor2 = torch.rand([batch_size * max_visit_num, max_diag_num])
# unsqueezed_tensor2 = tensor2.unsqueeze(-1)
# print(tensor2.shape)
# print(unsqueezed_tensor2.shape)

# print()
# print(tensor3.shape)
# print(tensor.shape)
# h = tensor3 * tensor
# output = torch.sum(h, dim=1)
# print(output.shape)


# visit_diag_embedding = torch.rand([batch_size, max_visit_num, 64])
# padding = torch.zeros((batch_size, 1, 64)).float()
# diag_keys = torch.cat([padding, visit_diag_embedding[:, :-1, :]], dim=1)
# print(visit_diag_embedding.shape)
# print(padding.shape)
# print(diag_keys.shape)

# seq = [1, 2, 3]
# print(seq[:-1])


# mask = torch.rand([batch_size * max_visit_num, 1, 1, 100]).repeat(1,2,100,1)
# print(mask.shape)

gcn_emb = torch.rand([batch_size, 64])
memory_padding = torch.rand([3, 64])
print(gcn_emb.shape, memory_padding.shape)
embedding = torch.cat([gcn_emb, memory_padding], dim=0)
print(embedding.shape)
import torch
from torch_scatter import scatter


class SoftCap(torch.nn.Module):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap


    def forward(self, inputs):
        outputs = inputs / self.cap
        outputs = torch.nn.functional.tanh(outputs)
        outputs = outputs * self.cap
        return outputs


    def __repr__(self):
        return f"{self.__class__.__name__}(cap={self.cap})"
        

class GraphSoftmax(torch.nn.Module):
    """
        1.  Reference: https://pytorch-geometric.readthedocs.io/en/2.3.1/_modules/torch_geometric/utils/softmax.html
        2.  Add `exp_dropout` so that we can remove the contributions of some neighbors while keeping
            the sum equal to 1.
        3.  Add `exp_rescale` to rescale the outputs of the exponentation function so that we can downscale the 
            contributions of some neighbors with an envelope function.
        4.  Add `softcap` to limit input logits to the range [- `softcap`, + `softcap`].
        5.  Add `eps` for numerical stability.
    """
    def __init__(self, eps=1e-16, exp_dropout=0.0, softcap=None):
        super().__init__()
        self.eps = eps
        self.exp_dropout = exp_dropout
        self.dropout = torch.nn.Dropout(exp_dropout) if self.exp_dropout > 0.0 else torch.nn.Identity()
        self.softcap = SoftCap(cap=softcap) if softcap is not None else torch.nn.Identity()


    def forward(
        self, 
        src, 
        index=None, 
        ptr=None, 
        num_nodes=None, 
        dim=0,
        exp_rescale=None
    ):
        src = self.softcap(src)
        if ptr is not None:
            raise NotImplementedError(
                "GraphSoftmax currently requires an index-based scatter, "
                "not a ptr-based segment."
            )
        elif index is not None:
            if num_nodes is None:
                num_nodes = int(index.max().item()) + 1 if index.numel() else 0
            N = num_nodes
            src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
            out = src - src_max.index_select(dim, index)
            out = out.exp()
            if exp_rescale is not None:
                out = out * exp_rescale
            out = self.dropout(out)
            out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + self.eps
            out_sum = out_sum.index_select(dim, index)
        else:
            raise NotImplementedError
        
        out = out / out_sum
        
        return out
    

    def extra_repr(self):
        return 'eps={}'.format(self.eps)

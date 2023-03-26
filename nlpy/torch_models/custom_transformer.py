import torch
from torch import nn
from .relative_attention import ClippedRelativeAttention

class TinyClippedRelativeAttentionTransformer(nn.Module):
    """
    This model is inspired by the cross-layer parameter sharing approach in
    ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations

    Parameters
    ----------
    d_model : int
        model dimension
    num_heads : int
        number of self-attention heads
        d_model / num_heads must be an integer
    num_layers : int
        number of layers
    clip_dist : int
        max relative distance for relative position embeddings
    max_len : int
        max sequence length of inputs
    share_att_params: bool, Default: False
        share attention parameters across layers
    share_lin_params : bool, Default: False
        share feed forward parameter across layers
    dropout : float
    """
    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        clip_dist,
        max_len,
        share_att_params=False,
        share_lin_params=False,
        dropout=0.1):

        super(TinyClippedRelativeAttentionTransformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.clip_dist = clip_dist
        self.max_len = max_len
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=d_model) for i in range(num_layers)])
        self.rga = nn.ModuleList(
            [ClippedRelativeAttention(
                d_model=d_model,
                num_heads=num_heads,
                clip_dist=clip_dist,
                max_len=max_len,
                dropout=dropout) for i in range(num_layers)])
        self.Er = nn.Parameter(torch.randn(clip_dist, self.rga[0].d_head))
        # share relative clipped embedding parameters
        for layer in self.rga:
            layer.Er = self.Er
        # if we want to share parameters for keys, queries and values across layers
        if share_att_params: self.set_shared_att_params()

        self.lin = nn.ModuleList(
            [nn.Linear(
                in_features=d_model,
                out_features=d_model) for i in range(num_layers)])
        # if we want to share parameter across feed forward layers
        if share_lin_params: self.set_share_lin_params()
    
    def set_shared_att_params(self):
        self.rga_shared_params_key_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.rga_shared_params_key_bias = nn.Parameter(torch.randn(self.d_model))
        self.rga_shared_params_query_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.rga_shared_params_query_bias = nn.Parameter(torch.randn(self.d_model))
        self.rga_shared_params_value_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.rga_shared_params_value_bias = nn.Parameter(torch.randn(self.d_model))
        for layer in self.rga:
            layer.key.weight = self.rga_shared_params_key_weight
            layer.key.bias = self.rga_shared_params_key_bias
            layer.query.weight = self.rga_shared_params_query_weight
            layer.query.bias = self.rga_shared_params_query_bias
            layer.value.weight = self.rga_shared_params_value_weight
            layer.value.bias = self.rga_shared_params_value_bias

    def set_share_lin_params(self):
        self.lin_shared_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.lin_shared_bias = nn.Parameter(torch.randn(self.d_model))
        for layer in self.lin:
            layer.weight = self.lin_shared_weight
            layer.bias = self.lin_shared_bias

    def forward(self, x):
        # x.shape is (batch_size, seq_len, d_model)
        for i in range(self.num_layers):
            x = x + self.rga[i](x)
            x = self.batch_norms[i](x.permute(0,2,1)).permute(0,2,1)
            x = self.lin[i](x)
        return x

if __name__ == '__main__':
    d_model = 16
    num_heads = 4
    num_layers = 2
    clip_dist = 7
    max_len = 60
    batch_size = 512
    inp = torch.randn(batch_size, max_len, d_model)
    model = TinyClippedRelativeAttentionTransformer(
        d_model, num_heads, num_layers, clip_dist, max_len, dropout=0.1)
    sum(p.numel() for p in model.parameters() if p.requires_grad)
    # import torch
    # from torch import nn
    # from nlpy.torch_models.custom_transformer import TinyClippedRelativeAttentionTransformer
    # should be:
    # self attention: num_layers * num_heads * 3 (q,k,v) *(d_model*(d_model/num_heads) weights + 4 biases) + clip_dist*(d_model/num_heads) clipped emb
    # linear: num_layers * (16*16 weights + 16 biases)
    # batchnorm: 2 layers * (16 means + 16 variances)
    # 2*4*3*(16*4 + 4) + 7*4 + 2 *(16*16 + 16) + 2*(16+16)
import torch
from torch import nn
from .relative_attention import RelativeClippedAttention

class TinyRelativeClippedAttentionTransformer(nn.Module):

    def __init__(self, d_model, num_heads, num_layers, clip_dist, max_len, share_att_params=False, dropout=0.1):
        super(TinyRelativeClippedAttentionTransformer, self).__init__()
        self.num_layers = num_layers
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=d_model) for i in range(num_layers)])
        self.rga = nn.ModuleList(
            [RelativeClippedAttention(
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
        if share_att_params:
            self.rga_shared_params_key_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.rga_shared_params_key_bias = nn.Parameter(torch.randn(d_model))
            self.rga_shared_params_query_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.rga_shared_params_query_bias = nn.Parameter(torch.randn(d_model))
            self.rga_shared_params_value_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.rga_shared_params_value_bias = nn.Parameter(torch.randn(d_model))
            for layer in self.rga:
                layer.key.weight = self.rga_shared_params_key_weight
                layer.key.bias = self.rga_shared_params_key_bias
                layer.query.weight = self.rga_shared_params_query_weight
                layer.query.bias = self.rga_shared_params_query_bias
                layer.value.weight = self.rga_shared_params_value_weight
                layer.value.bias = self.rga_shared_params_value_bias
        self.lin = nn.ModuleList(
            [nn.Linear(
                in_features=d_model,
                out_features=d_model) for i in range(num_layers)])
    
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
    model = TinyRelativeClippedAttentionTransformer(
        d_model, num_heads, num_layers, clip_dist, max_len, dropout=0.1)
    sum(p.numel() for p in model.parameters() if p.requires_grad)
    # import torch
    # from torch import nn
    # from nlpy.torch_models.custom_transformer import TinyRelativeClippedAttentionTransformer
    # should be:
    # self attention: num_layers * num_heads * 3 (q,k,v) *(d_model*(d_model/num_heads) weights + 4 biases) + clip_dist*(d_model/num_heads) clipped emb
    # linear: 2 layers * (16*16 weights + 16 biases)
    # batchnorm: 2 layers * (16 means + 16 variances)
    # 2*4*3*(16*4 + 4) + 7*4 + 2 *(16*16 + 16) + 2*(16+16)
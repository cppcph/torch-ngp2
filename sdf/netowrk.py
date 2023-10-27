import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder


class SDFNetwork(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf

        self.encoder, self.in_dim = get_encoder(encoding)

        backbone = []

        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            elif l in self.skips:
                in_dim = self.hidden_dim + self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = self.hidden_dim
            
            backbone.append(nn.Linear(in_dim, out_dim, bias=False))

        self.backbone = nn.ModuleList(backbone)

    
    def forward(self, x):
        # x: [B, 3]

        x = self.encoder(x)

        h = x
        for l in range(self.num_layers):
            if l in self.skips:
                h = torch.cat([h, x], dim=-1)
            h = self.backbone[l](h)
            if l != self.num_layers - 1:
                h = F.elu(h, inplace=False)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h
    
class SDFNetworkWithSubspaceInput(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=3,
                 num_layers_pre=0,
                 subspace_size=4,
                 hidden_dim=64,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.num_layers_pre = num_layers_pre
        self.hidden_dim = hidden_dim

        self.encoder, self.in_dim = get_encoder(encoding)
        backbone = []
        preencoder=[]

        if num_layers_pre > 0:
            for l in range(num_layers_pre):
                if l == 0:
                    in_dim = 3 + subspace_size
                else:
                    in_dim = self.hidden_dim
                
                if l == num_layers_pre - 1:
                    out_dim = 3
                else:
                    out_dim = self.hidden_dim
                
                preencoder.append(nn.Linear(in_dim, out_dim, bias=False))

            self.preencoder= nn.ModuleList(preencoder)

        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + subspace_size
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = self.hidden_dim
            
            backbone.append(nn.Linear(in_dim, out_dim, bias=False))

        self.backbone = nn.ModuleList(backbone)

    
    def forward(self, x, subspace):
        # x: [B, 3]
        if self.num_layers_pre > 0:
            h = torch.cat([x, subspace], dim=-1)
            for l in range(self.num_layers_pre):
                h = self.preencoder[l](h)
                if l != self.num_layers_pre - 1:
                    h = F.elu(h, inplace=False)
                # else:
                #     #resitrict the output to be in the range of [-1,1]
                #     h = torch.tanh(h)
            preencoder_output = h
        else:
            preencoder_output=  x

        x = self.encoder(preencoder_output)

        h = torch.cat([x, subspace], dim=-1)
        for l in range(self.num_layers):
            h = self.backbone[l](h)
            if l != self.num_layers - 1:
                h = F.elu(h, inplace=False)

        return h,preencoder_output
    
class SDFNetworkWithSubspaceInputOnlyForPreencoder(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=3,
                 num_layers_pre=0,
                 subspace_size=4,
                 hidden_dim=64,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.num_layers_pre = num_layers_pre
        self.hidden_dim = hidden_dim

        self.encoder, self.in_dim = get_encoder(encoding)
        backbone = []
        preencoder=[]

        if num_layers_pre > 0:
            for l in range(num_layers_pre):
                if l == 0:
                    in_dim = 3 + subspace_size
                else:
                    in_dim = self.hidden_dim
                
                if l == num_layers_pre - 1:
                    out_dim = 3
                else:
                    out_dim = self.hidden_dim
                
                preencoder.append(nn.Linear(in_dim, out_dim, bias=False))

            self.preencoder= nn.ModuleList(preencoder)

        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim 
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = self.hidden_dim
            
            backbone.append(nn.Linear(in_dim, out_dim, bias=False))

        self.backbone = nn.ModuleList(backbone)

    
    def forward(self, x, subspace):
        # x: [B, 3]
        if self.num_layers_pre > 0:
            h = torch.cat([x, subspace], dim=-1)
            for l in range(self.num_layers_pre):
                h = self.preencoder[l](h)
                if l != self.num_layers_pre - 1:
                    h = F.elu(h, inplace=False)
                # else:
                #     #resitrict the output to be in the range of [-1,1]
                #     h = torch.tanh(h)
            preencoder_output = h
        else:
            preencoder_output = x

        x = self.encoder(preencoder_output)

        h = x
        for l in range(self.num_layers):
            h = self.backbone[l](h)
            if l != self.num_layers - 1:
                h = F.elu(h, inplace=False)

        return h,preencoder_output
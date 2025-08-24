import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModel,AutoConfig,DebertaV2Config
#mish activation
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))


from torch.nn.parameter import Parameter
def gem(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.gamma=nn.Parameter(torch.tensor(100.0))

    def forward(self, q, k, v, mask=None, attn_mask=None):

        attn = torch.matmul(q, k.transpose(2, 3))/ self.temperature
        to_plot=attn[0,0].detach().cpu().numpy()
        # plt.imshow(to_plot)
        # plt.show()
        # exit()

        #exit()
        if mask is not None:
            #attn = attn.masked_fill(mask == 0, -1e9)
            #attn = attn#*self.gamma
            attn = attn+mask*self.gamma
        if attn_mask is not None:
            # print(attn.shape)
            # print(attn_mask.shape)
            # attn = attn+attn_mask
            #attn=attn.float().masked_fill(attn_mask == 0, float('-inf'))
            attn=attn.float().masked_fill(attn_mask == 0, float('-1e-9'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        # print(attn[0,0])
        # to_plot=attn[0,0].detach().cpu().numpy()
        # with open('mat.txt','w+') as f:
        #     for vector in to_plot:
        #         for num in vector:
        #             f.write('{:04.3f} '.format(num))
        #         f.write('\n')
        # plt.imshow(to_plot)
        # plt.show()
        # exit()
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None,src_mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask  # For head axis broadcasting

        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        if src_mask is not None:
            src_mask=src_mask[:,:q.shape[2]].unsqueeze(-1).float()
            # q=q+src_mask
            # k=k+src_mask
            # print(src_mask.shape)
            # print(src_mask[0])
            attn_mask=torch.matmul(src_mask,src_mask.permute(0,2,1))#.long()
            #attn_mask=attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
            attn_mask=attn_mask.unsqueeze(1)
            # print(attn_mask.shape)
            # exit()
            # print(src_mask.shape)
            #to_plot=attn_mask[1].squeeze().detach().cpu().numpy()
            #plt.imshow(to_plot)
            #plt.show()
            # exit()
            # exit()
            # src_mask
            # src_mask
            #print(q[0,0,:,0])
        #exit()
            q, attn = self.attention(q, k, v, mask=mask,attn_mask=attn_mask)
        else:
            q, attn = self.attention(q, k, v, mask=mask)
        #print(attn.shape)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        #print(q.shape)
        #exit()
        # q = self.dropout(self.fc(q))
        # q += residual

        #q = self.layer_norm(q)

        return q, attn

class ConvDebertaTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, k = 3):
        super(ConvDebertaTransformerEncoderLayer, self).__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)


        config_model = DebertaV2Config()

        config_model.hidden_size=d_model
        config_model.intermediate_size=d_model*4
        config_model.num_attention_heads=nhead
        config_model.num_hidden_layers=1
        config_model.vocab_size=4
        model=AutoModel.from_config(config_model)

        self.encoder=model.encoder

        #self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)
        # self.mask_conv1 = nn.Conv2d(1,d_model,k)
        # self.mask_activation1=nn.ReLU()
        # self.mask_conv2 = nn.Conv2d(d_model,nhead,1)
        self.mask_conv1 = nn.Sequential(nn.Conv2d(nhead//4,nhead//4,k),
                                        nn.BatchNorm2d(nhead//4),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(nhead//4,nhead,1),
                                        # nn.BatchNorm2d(nhead),
                                        # nn.ReLU(inplace=True),
                                        )
        self.mask_deconv = nn.Sequential(nn.ConvTranspose2d(nhead,nhead//4,k),
                                        nn.BatchNorm2d(nhead//4),
                                        nn.Sigmoid()
                                        )

        #
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)


        self.activation = nn.ReLU()

        self.conv=nn.Conv1d(d_model,d_model,k,padding=0)
        self.deconv=nn.ConvTranspose1d(d_model,d_model,k)

    def forward(self, src , mask=None, src_mask=None):
        res = src
        #print(src.shape)
        #exit()
        src = self.norm3(self.conv(src.permute(0,2,1)).permute(0,2,1))
        if mask is not None:
            mask_res=mask
            mask = self.mask_conv1(mask)

        src = self.encoder(src,attention_mask=torch.ones(src.shape[0],src.shape[1]).cuda(),return_dict=False)[0]
        attention_weights = None
        # print(src.shape)
        # exit()

        src = res + self.dropout3(self.deconv(src.permute(0,2,1)).permute(0,2,1))
        # src = self.norm4(src)
        if mask is not None:
            mask = self.mask_deconv(mask)+mask_res
            return src,attention_weights,mask
        else:
            return src,attention_weights

class ConvTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, k = 3):
        super(ConvTransformerEncoderLayer, self).__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)
        # self.mask_conv1 = nn.Conv2d(1,d_model,k)
        # self.mask_activation1=nn.ReLU()
        # self.mask_conv2 = nn.Conv2d(d_model,nhead,1)
        self.mask_conv1 = nn.Sequential(nn.Conv2d(nhead//4,nhead//4,k),
                                        nn.BatchNorm2d(nhead//4),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(nhead//4,nhead,1),
                                        # nn.BatchNorm2d(nhead),
                                        # nn.ReLU(inplace=True),
                                        )
        self.mask_deconv = nn.Sequential(nn.ConvTranspose2d(nhead,nhead//4,k),
                                        nn.BatchNorm2d(nhead//4),
                                        nn.Sigmoid()
                                        )
        # self.mask_activation2=nn.ReLU()
        # self.mask_conv3 = nn.Conv2d(d_model//2,nhead,1)
        #torch.nn.init.ones_(self.mask_conv.weight)
        #self.mask_conv.weight.requires_grad=False
        #self.mask_conv.weight.requires_grad=False
        #self.mask_conv.requires_grad=False
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        #self.dropout4 = nn.Dropout(dropout)


        self.activation = nn.ReLU()

        self.conv=nn.Conv1d(d_model,d_model,k,padding=0)
        self.deconv=nn.ConvTranspose1d(d_model,d_model,k)

    def forward(self, src , mask=None, src_mask=None):
        res = src
        #print(src.shape)
        #exit()
        src = self.norm3(self.conv(src.permute(0,2,1)).permute(0,2,1))
        if mask is not None:
            mask_res=mask
            mask = self.mask_conv1(mask)
        #mask = self.mask_activation1(mask)
        #mask = self.mask_conv2(mask)
        # mask = self.mask_activation2(mask)
        # mask = self.mask_conv3(mask)
        src2,attention_weights = self.self_attn(src, src, src, mask=mask, src_mask=src_mask)
        #src3,_ = self.self_attn(src, src, src, mask=None)
        #src2=src2+src3
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = res + self.dropout3(self.deconv(src.permute(0,2,1)).permute(0,2,1))
        src = self.norm4(src)
        if mask is not None:
            mask = self.mask_deconv(mask)+mask_res
            return src,attention_weights,mask
        else:
            return src,attention_weights

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)





# class Degradation_Model(nn.Module):
#
#     def __init__(self, ntoken, nclass, ninp, nhead, nhid, nlayers, kmer_aggregation, kmers, stride=1,dropout=0.5,pretrain=False,return_aw=False):
#         super(Degradation_Model, self).__init__()
#         self.model_type = 'Transformer'
#         self.src_mask = None
#         #self.pos_encoder = PositionalEncoding(ninp, dropout)
#         self.kmers=kmers
#         #if self.ngrams!=None:
#         self.transformer_encoder = []
#         for i in range(nlayers):
#             self.transformer_encoder.append(ConvTransformerEncoderLayer(ninp, nhead, nhid, dropout, k=kmers[0]-i))
#         self.transformer_encoder= nn.ModuleList(self.transformer_encoder)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         #self.projection = nn.Linear(ninp*3, ninp)
#         #self.directional_encoder = nn.Embedding(3, ninp//8)
#         self.ninp = ninp
#         self.decoder = nn.Linear(ninp,nclass)
#         self.mask_dense=nn.Conv2d(4,nhead//4,1)
#         # self.recon_decoder = LinearDecoder(ntoken,ninp,dropout,pool=False)
#         # self.error_decoder = LinearDecoder(2,ninp,dropout,pool=False)
#
#         self.return_aw=return_aw
#         self.pretrain=pretrain
#
#
#         self.pretrain_decoders=nn.ModuleList()
#         self.pretrain_decoders.append(nn.Linear(ninp,4))
#         #self.pretrain_decoders.append(nn.Linear(ninp,3))
#         #self.pretrain_decoders.append(nn.Linear(ninp,7))
#
#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
#
#     def forward(self, src, mask=None,src_mask=None):
#         B,L=src.shape
#         src = src
#         src = self.encoder(src).reshape(B,L,-1)
#         #src =self.projection(src)
#         #src = self.pos_encoder(src.permute(1,0,2)).permute(1,0,2)
#
#         #mask=mask.unsqueeze(1)
#         mask=self.mask_dense(mask)
#         for i,layer in enumerate(self.transformer_encoder):
#             if src_mask is not None:
#                 src,attention_weights_layer,mask=layer(src, mask, src_mask[:,i])
#             else:
#                 src,attention_weights_layer,mask=layer(src, mask)
#             #attention_weights.append(attention_weights_layer)
#         #attention_weights=torch.stack(attention_weights).permute(1,0,2,3)
#         encoder_output = src
#         #print(deconved.shape)
#         #print(encoder_output.shape)
#         output = self.decoder(encoder_output)
#         # recon_src = self.recon_decoder(encoder_output)
#         # error_src = self.error_decoder(encoder_output)
#         if self.pretrain:
#             ae_outputs=[]
#             for decoder in self.pretrain_decoders:
#                 ae_outputs.append(decoder(encoder_output))
#             return ae_outputs
#         else:
#             if self.return_aw:
#                 return output,attention_weights_layer
#             else:
#                 return output



# class Degradation_Model(nn.Module):
#
#     def __init__(self, ntoken, nclass, ninp, nhead, nhid, nlayers, kmer_aggregation, kmers, stride=1,dropout=0.5,pretrain=False,return_aw=False):
#         super(Degradation_Model, self).__init__()
#         self.model_type = 'Transformer'
#         self.src_mask = None
#         #self.pos_encoder = PositionalEncoding(ninp, dropout)
#         self.kmers=kmers
#         #if self.ngrams!=None:
#
#         self.transformer_encoder = []
#         for i in range(nlayers):
#             self.transformer_encoder.append(ConvTransformerEncoderLayer(ninp, nhead, nhid, dropout, k=kmers[0]-i))
#         self.transformer_encoder= nn.ModuleList(self.transformer_encoder)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         self.projection = nn.Linear(ninp*3, ninp)
#         #self.directional_encoder = nn.Embedding(3, ninp//8)
#         self.ninp = ninp
#         self.decoder = nn.Linear(ninp,nclass)
#         self.mask_dense=nn.Conv2d(4,nhead//4,1)
#         # self.recon_decoder = LinearDecoder(ntoken,ninp,dropout,pool=False)
#         # self.error_decoder = LinearDecoder(2,ninp,dropout,pool=False)
#
#         self.return_aw=return_aw
#         self.pretrain=pretrain
#
#
#         self.pretrain_decoders=nn.ModuleList()
#         self.pretrain_decoders.append(nn.Linear(ninp,4))
#         self.pretrain_decoders.append(nn.Linear(ninp,3))
#         self.pretrain_decoders.append(nn.Linear(ninp,7))
#
#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
#
#     def forward(self, src, mask=None):
#         B,L,_=src.shape
#         src = src
#         src = self.encoder(src).reshape(B,L,-1)
#         src =self.projection(src)
#         #src = self.pos_encoder(src.permute(1,0,2)).permute(1,0,2)
#
#         #mask=mask.unsqueeze(1)
#         mask=self.mask_dense(mask)
#         for layer in self.transformer_encoder:
#             src,attention_weights_layer,mask=layer(src, mask)
#             #attention_weights.append(attention_weights_layer)
#         #attention_weights=torch.stack(attention_weights).permute(1,0,2,3)
#         attention_weights_layer=attention_weights_layer.mean(1)
#         encoder_output = src
#         #print(deconved.shape)
#         #print(encoder_output.shape)
#         output = self.decoder(encoder_output)
#         # recon_src = self.recon_decoder(encoder_output)
#         # error_src = self.error_decoder(encoder_output)
#         if self.pretrain:
#             ae_outputs=[]
#             for decoder in self.pretrain_decoders:
#                 ae_outputs.append(decoder(encoder_output))
#             return ae_outputs
#         else:
#             if self.return_aw:
#                 return output,attention_weights_layer
#             else:
#                 return output



class NucleicTransformer(nn.Module):

    def __init__(self, ntoken, nclass, ninp, nhead, nhid, nlayers, k, cai_vector, cai_weight, stride=1,dropout=0.1,return_aw=False,clip=(-1,1),new_head=False, use_deberta_attention=False):
        super(NucleicTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.k=k
        #if self.ngrams!=None:
        self.transformer_encoder = []
        for i in range(nlayers):
            if use_deberta_attention:
                self.transformer_encoder.append(ConvDebertaTransformerEncoderLayer(ninp, nhead, nhid, dropout, k=k-i))
            else:
                self.transformer_encoder.append(ConvTransformerEncoderLayer(ninp, nhead, nhid, dropout, k=k-i))
        self.transformer_encoder= nn.ModuleList(self.transformer_encoder)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.start_conv = nn.ConvTranspose1d(ninp,ninp,3,stride=3)
        self.start_norm = nn.LayerNorm(ninp)
        self.ninp = ninp
        self.cai_vector = cai_vector.reshape(1,1,64)
        self.cai_weight = cai_weight
        self.end_conv = nn.Conv1d(ninp,ninp,3,stride=3)
        self.end_norm = nn.LayerNorm(ninp)

        self.output_layer=nn.Linear(ninp,nclass)

        self.new_head=new_head
        if self.new_head:
            self.output_layer2=nn.Linear(ninp,nclass)
            self.output_layer3=nn.Linear(ninp,nclass)

        #self.sigmoid=nn.Sigmoid()
        self.return_aw=return_aw
        self.clip=clip



    def forward(self, src, mask=None):
        B,L=src.shape
        src = src
        src = self.encoder(src).reshape(B,L,-1)

        #print(src.shape)
        #src = self.start_norm(self.start_conv(src.permute(0,2,1)).permute(0,2,1))

        src = self.pos_encoder(src.permute(1,0,2)).permute(1,0,2)
        # print(src.shape)
        # exit()

        for layer in self.transformer_encoder:
            src,attention_weights_layer=layer(src)

        #src = self.end_norm(self.end_conv(src.permute(0,2,1)).permute(0,2,1))
        output=self.output_layer(src)+self.cai_vector*self.cai_weight#.clip(self.clip[0],self.clip[1])

        if self.new_head:
            new_head_output=self.output_layer2(src)+self.cai_vector*self.cai_weight
            new_head_output2=self.output_layer3(src)+self.cai_vector*self.cai_weight
        if self.new_head:
            if self.return_aw:
                return output, new_head_output, attention_weights_layer
            else:
                return output, new_head_output, new_head_output2
        else:
            if self.return_aw:
                return output, attention_weights_layer
            else:
                return output

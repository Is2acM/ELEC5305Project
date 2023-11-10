'''
Script for defining the network. (ViT)
------------------------------------------- 
Reference: https://huggingface.co/google/vit-base-patch16-224-in21k (Pretrained weights)
           https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py
           https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.vit_b_16 
'''
from torch import nn, Tensor
import torch
import math
from collections import OrderedDict
from datasets import classes

class Config:
    '''Configuring the parameters of the ViT'''
    def __init__(self) -> None:
        # default use square
        self.patch_size = 16
        self.image_size = 224
        self.channel = 3
        self.train_batch = 8
        self.hidden_feature = 768
        self.hidden_heads = 12
        # hidden layer
        self.hidden_layer = 12
        # disabled as implemented in transformer
        self.drop_out_prob = 0
        self.drop_out_attention_ouput = 0
        self.drop_hidden_prob = 0
        self.layer_norm_eps = 1e-12 # avoid dividen by 0
        self.intermediate_size = 3072 # hidden layer in MLP, extract feature from attention
        # define classification
        self.class_num = len(classes)
        self.class_hidden_feature = 128

config = Config()

class VitPatchEmbeddings(nn.Module):
    
    def __init__(self, config:Config):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_num = self.image_size // self.patch_size
        self.patch_num *= self.patch_num # square
        self.channel = config.channel
        self.projection = nn.Conv2d(
            self.channel, 
            config.hidden_feature,
            # TODO:use patch size to conf, may be pooling can be used?
            kernel_size=self.patch_size, 
            stride=self.patch_size
            )

    def forward(self, img: Tensor):
        # TODO: Just assume picture is ready as config, skip for checking
        patch_out = self.projection(img)
        # output [flat to line, blocks, features], maybe batch need to fit
        return patch_out.flatten(2).transpose(1,2)

class VitAttention(nn.Module):

    def __init__(self, config:Config):
        super(VitAttention, self).__init__()
        hidden_size = config.hidden_feature
        self.head_num = config.hidden_heads
        # make sure hiddensize is mulitple of heads
        self.head_size = int(hidden_size / self.head_num)
        self.hidden_size = hidden_size
        # because the hidden maybe not multiple to headnum
        self.full_head_size = self.head_size * self.head_num
        # implement qkv
        self.query = nn.Linear(
            hidden_size, 
            hidden_size,
            True # with qkv bias
        )
        self.key = nn.Linear(
            hidden_size, 
            hidden_size,
            True # with qkv bias
        )
        self.value = nn.Linear(
            hidden_size, 
            hidden_size,
            True # with qkv bias
        )
        self.dropout = nn.Dropout(
            0 # copy from the pre model
        )
        self.norm_before = nn.LayerNorm(config.hidden_feature, eps=config.layer_norm_eps)
        # output post deal
        self.output_dense = nn.Linear(config.hidden_feature, config.hidden_feature)
        self.output_dropout = nn.Dropout(config.drop_out_attention_ouput)

    def expand_out_to_head_feature(self, feature:Tensor):
        # 3-dim to 4-dim
        new_shape = feature.size()[:-1] + (self.head_num, self.head_size)
        feature = feature.view(new_shape)
        # to batch, head, pos, feature
        return feature.permute(0, 2, 1, 3)

    def forward(self, feature:Tensor):
        normed_feature = self.norm_before(feature)
        out_q = self.expand_out_to_head_feature(
            self.query(normed_feature)
        )
        out_k = self.expand_out_to_head_feature(
            self.key(normed_feature)
        )
        out_v = self.expand_out_to_head_feature(
            self.value(normed_feature)
        )
        # calcuate score A = Q x T(K)
        attention_score = torch.matmul(out_q, out_k.transpose(-1, -2))
        # / sqrt(d)
        attention_score = attention_score / math.sqrt(self.head_size)
        # nomalize with softmax, for the features in each head
        attention_prob = nn.functional.softmax(attention_score, dim=-1)
        # use dropout, partially
        attention_prob = self.dropout(attention_prob)

        # add context_layer, create b
        context_layer = torch.matmul(attention_prob, out_v)
        # revert the seq to batch, pos, head, feature
        # contiguous force the matrix sequential, improve performance
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # merge the head and headfeature together
        # seq to batch, pos, feature
        new_context_shape = context_layer.size()[:-2] + (self.full_head_size,)
        context_layer = context_layer.view(new_context_shape)
        # if wanna to output attention_probs( for heatmap I thought) output attention_probs here
        
        # Add origianl feature and norm
        attention_out = self.output_dense(context_layer)
        attention_out = self.output_dropout(attention_out)

        return attention_out
    
class VitLayer(nn.Module):

    def __init__(self, config:Config):
        super().__init__()
        self.attention = VitAttention(config)
        self.norm_after = nn.LayerNorm(config.hidden_feature, eps=config.layer_norm_eps)
        self.dense_intermediate = nn.Linear(config.hidden_feature, config.intermediate_size)
        self.active_intermediate = nn.functional.gelu
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_feature)
        self.output_dropout = nn.Dropout(config.drop_hidden_prob)

    def forward(self, hidden_feature:Tensor):
        attention_out = self.attention(hidden_feature)
        hidden_feature = attention_out + hidden_feature
        # norm after add hidden feature and attention output
        # intermediate
        layer_output = self.norm_after(hidden_feature)
        layer_output = self.dense_intermediate(layer_output)
        layer_output = self.active_intermediate(layer_output)

        # output -> 768
        layer_output = self.output_dense(layer_output)
        layer_output = self.output_dropout(layer_output)
        # add the input
        layer_output = layer_output + hidden_feature
        return layer_output
    
class VitEmbeddings(nn.Module):

    def __init__(self, config:Config):
        super().__init__()
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_feature)
        )
        self.patch_embeddings = VitPatchEmbeddings(config)
        self.position_embeddings = nn.Parameter(
            torch.randn(
                1, 
                self.patch_embeddings.patch_num+1, 
                config.hidden_feature
            )
        )
        self.dropout = nn.Dropout(config.drop_out_prob)
        
    def forward(self, img:Tensor):
        batch, channel, height, weight = img.shape
        embeddings = self.patch_embeddings(img)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # Add position embedding
        embeddings = embeddings + self.position_embeddings
        # Optional dropout
        embeddings = self.dropout(embeddings)
        return embeddings
    
class VitNetwork(nn.Module):
    
    def __init__(self, config:Config):
        super().__init__()
        # Add class embedding, used for global embedding
        self.embeddings = VitEmbeddings(config)
        # Add layers
        self.layers = nn.ModuleList(
            [VitLayer(config) for _ in range(config.hidden_layer)]
        )
        self.layernorm = nn.LayerNorm(
            config.hidden_feature,
            eps = config.layer_norm_eps
        )

    def forward(self, img:Tensor):
        # prepare cls for batch
        embeddings = self.embeddings(img)
        # Connect the layers one by one
        hidden_feature = embeddings
        for layer in self.layers:
            hidden_feature = layer(hidden_feature)
        sequence_output = self.layernorm(hidden_feature)
        # pooling means simpliy taking hidden from first token
        return sequence_output

class FeatureClassifier(nn.Module):
    '''Added two fully connected layers for image classification'''
    
    def __init__(self, config:Config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_feature, config.class_hidden_feature)
        self.fc2 = nn.Linear(config.class_hidden_feature, config.class_num)
        
    def forward(self, feature:Tensor):
        hidden_class = self.fc1(feature) # Mel and MFCCs
        logits = self.fc2(nn.functional.tanh(hidden_class))
        return logits
    
class VitClassifier(nn.Module):
    '''Combine the model'''
    def __init__(self, config:Config):
        super().__init__()
        self.vit = VitNetwork(config)
        self.classifier = FeatureClassifier(config)

    def forward(self, img:Tensor):
        hidden_feature = self.vit(img)
        # cls head is the class feature
        class_feature = hidden_feature[:, 0, :]
        classify = self.classifier(class_feature)
        return classify

def remap_weights(orig:OrderedDict):
    '''Function for remaping the weights'''
    out = OrderedDict()
    for k, v in orig.items():
        if k.startswith('embeddings'):
            out[k] = v
        elif k.startswith('encoder'):
            parts = k.split('.')
            layer_idx = parts[2]
            if (parts[3], parts[4]) == ('attention', 'attention'):
                nk = f'layers.{layer_idx}.attention.{".".join(parts[-2:])}'
            if (parts[3], parts[4]) == ('attention', 'output'):
                nk = f'layers.{layer_idx}.attention.output_dense.{parts[-1]}'
            elif parts[3] == 'intermediate':
                nk = f'layers.{layer_idx}.dense_intermediate.{parts[-1]}'
            elif parts[3] == 'output':
                nk = f'layers.{layer_idx}.output_dense.{parts[-1]}'
            elif parts[3] == 'layernorm_before':
                nk = f'layers.{layer_idx}.attention.norm_before.{parts[-1]}'
            elif parts[3] == 'layernorm_after':
                nk = f'layers.{layer_idx}.norm_after.{parts[-1]}'
            out[nk] = v
        elif k.startswith('layernorm'):
            out[k] = v
    return out

def Network(train = False):
    if train == True:
        ## Load the google pretrain vit weights
        model = VitClassifier(config)
        ## Rename layers in pretrain weights and apply that to the ViT
        re_dict = remap_weights(torch.load('./google_vit_pretrain.pth'))
        ## Using pretrained ViT, and fine-tunning it for training
        model.vit.load_state_dict(re_dict)
    else:
        model = VitClassifier(config)
    return model


import copy
import logging
import math
import torch
from torch import nn
import timm
from torch.nn import functional as F

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        self.use_RP=False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn=input
                #inn=torch.bmm(input[:,0:100].unsqueeze(-1), input[:,0:100].unsqueeze(-2)).flatten(start_dim=1) #interaction terms instead of RP
            out = F.linear(inn,self.weight)

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


def get_convnet(args, pretrained=False):

    name = args["convnet_type"].lower()
    #Resnet
    if name=="pretrained_resnet50":
        from resnet import resnet50
        model=resnet50(pretrained=True,args=args)
        return model.eval()
    elif name=="pretrained_resnet152":
        from resnet import resnet152
        model=resnet152(pretrained=True,args=args)
        return model.eval()
    elif name=="vit_base_patch32_224_clip_laion2b":
        #note: even though this is "B/32" it has nearly the same num params as the standard ViT-B/16
        model=timm.create_model("vit_base_patch32_224_clip_laion2b", pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    
    #NCM or NCM w/ Finetune
    elif name=="pretrained_vit_b16_224" or name=="vit_base_patch16_224":
        model=timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    elif name=="pretrained_vit_b16_224_in21k" or name=="vit_base_patch16_224_in21k":
        model=timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    
    # SSF 
    elif '_ssf' in name:
        if args["model_name"]=="ssf":
            from petl import vision_transformer_ssf #registers vit_base_patch16_224_ssf
            if name=="pretrained_vit_b16_224_ssf":
                model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
                model.out_dim=768
            elif name=="pretrained_vit_b16_224_in21k_ssf":
                model=timm.create_model("vit_base_patch16_224_in21k_ssf",pretrained=True, num_classes=0)
                model.out_dim=768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    
    # VPT
    elif '_vpt' in name:
        if args["model_name"]=="vpt":
            from petl.vpt import build_promptmodel
            if name=="pretrained_vit_b16_224_vpt":
                basicmodelname="vit_base_patch16_224" 
            elif name=="pretrained_vit_b16_224_in21k_vpt":
                basicmodelname="vit_base_patch16_224_in21k"
            
            #print("modelname,",name,"basicmodelname",basicmodelname)
            VPT_type="Deep"
            #if args["vpt_type"]=='shallow':
            #    VPT_type="Shallow"
            Prompt_Token_num=5#args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname,  Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim=768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_adapter' in name:
        ffn_num=64#args["ffn_num"]
        if args["model_name"]=="adapter" :
            from petl import vision_transformer_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name=="pretrained_vit_b16_224_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name=="pretrained_vit_b16_224_in21k_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
        
    # elif '_lora' in name:
    #     if args["model_name"] == "lora":
    #         from petl import vision_transformer_lora
    #         #from vision_transformer_lora import vit_base_patch16_224_lora, vit_base_patch16_224_in21k_lora
            
    #         # 从args获取LoRA的超参数
    #         r = int(args["lora_r"])  
    #         lora_alpha = int(args["lora_alpha"])  
    #         logging.info(type(r))
    #         logging.info(f'LoRA r: {r}, alpha: {lora_alpha}')
    #         if name == "pretrained_vit_b16_224_lora":
    #             model = vision_transformer_lora.vit_base_patch16_224_lora(
    #                 pretrained=True,
    #                 r=r,
    #                 lora_alpha=lora_alpha,
    #                 num_classes=0
    #             )
    #             model.out_dim = 768
    #         elif name == "pretrained_vit_b16_224_in21k_lora":
    #             model = vision_transformer_lora.vit_base_patch16_224_in21k_lora(
    #                 pretrained=True,
    #                 r=r,
    #                 lora_alpha=lora_alpha,
    #                 num_classes=0
    #             )
    #             model.out_dim = 768
    #         else:
    #             raise NotImplementedError(f"Unknown LoRA model type: {name}")
                
    #         # 验证LoRA参数的训练状态
    #         trainable_params = 0
    #         all_params = 0
    #         for name, param in model.named_parameters():
    #             all_params += param.numel()
    #             if param.requires_grad:
    #                 trainable_params += param.numel()
    #         logging.info(f"LoRA trainable parameters: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}% of all parameters)")
            
    #         return model
    #     else:
    #         raise NotImplementedError("Inconsistent model name and model type")
    
    elif '_peft' in name:
        if args["model_name"] in ["lora", "loha", "lokr"]:
            from petl import vision_transformer_peft
            
            # 获取PEFT的通用超参数
            r = int(args["peft_r"])  
            alpha = int(args["peft_alpha"])
            peft_type = args["model_name"]  # 'lora', 'loha', 或 'lokr'
            
            # LoKR特有的参数
            #factor = int(args.get("lokr_factor", -1)) if peft_type == 'lokr' else -1

            logging.info(f'PEFT type: {peft_type}, r: {r}, alpha: {alpha}')


            if name == "pretrained_vit_b16_224_peft":

                model = vision_transformer_peft.vit_base_patch16_224_peft(
                    pretrained=True,
                    peft_type=peft_type,
                    r=r,
                    alpha=alpha,
                    #factor=factor if peft_type == 'lokr' else None,
                    num_classes=0
                )
                model.out_dim = 768
            elif name in "pretrained_vit_b16_224_in21k_peft":
                model = vision_transformer_peft.vit_base_patch16_224_in21k_peft(
                    pretrained=True,
                    peft_type=peft_type,
                    r=r,
                    alpha=alpha,
                    #factor=factor if peft_type == 'lokr' else None,
                    num_classes=0
                )
                model.out_dim = 768
            else:
                raise NotImplementedError(f"Unknown PEFT model type: {name}")
                
            # 验证PEFT参数的训练状态
            trainable_params = 0
            all_params = 0
            for name, param in model.named_parameters():
                all_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            logging.info(f"{peft_type.upper()} trainable parameters: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}% of all parameters)")
            
            return model
        
        # elif args["model_name"] in ["2lora"]:
        #     from petl import vision_transformer_peft
        #     # 获取PEFT的通用超参数
        #     r = int(args["peft_r"])  
        #     alpha = int(args["peft_alpha"])
        #     peft_type = 'lora'  # 'lora', 'loha', 或 'lokr'

        #     if name in "vit_base_patch16_224_in21k_multilora":
        #         model = vision_transformer_peft.vit_base_patch16_224_in21k_peft(
        #             pretrained=True,
        #             num_loras=2,
        #             r=r,
        #             alpha=alpha,
        #             #factor=factor if peft_type == 'lokr' else None,
        #             num_classes=0
        #         )
        #         model.out_dim = 768

        #     else:
        #         raise NotImplementedError(f"Unknown PEFT model type: {name}")
            
        #     trainable_params = 0
        #     all_params = 0
        #     for name, param in model.named_parameters():
        #         all_params += param.numel()
        #         if param.requires_grad:
        #             trainable_params += param.numel()
        #     logging.info(f"{peft_type.upper()} trainable parameters: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}% of all parameters)")
            
        #     return model
        
        else:
            raise NotImplementedError("Inconsistent model name and model type")
        
    else:
        raise NotImplementedError("Unknown type {}".format(name))
    
    
class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

class ResNetCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.transfer = SimpleLinear(self.feature_dim, self.feature_dim)

        self.linear_fc = UnifiedLinear(self.feature_dim, args["init_cls"]).cuda()
    
    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

        transfer = SimpleLinear(self.feature_dim, self.feature_dim) 
        
        transfer.weight = nn.Parameter(torch.eye(self.feature_dim))
        transfer.bias = nn.Parameter(torch.zeros(self.feature_dim))
        del self.transfer
        self.transfer = transfer

        # 更新线性分类器的输出维度
        if hasattr(self, 'linear_fc'):
            in_features = self.linear_fc.in_features
            out_features = nb_classes
            new_linear_fc = UnifiedLinear(in_features, out_features).cuda()
            with torch.no_grad():
                if self.linear_fc.out_features < out_features:
                    # 扩展原始权重以适应新类别
                    new_weight = torch.cat(
                        [self.linear_fc.linear.weight.data, torch.zeros(out_features - self.linear_fc.out_features, in_features).cuda()],
                        dim=0
                    )
                    new_linear_fc.linear.weight = nn.Parameter(new_weight)
                    new_linear_fc.linear.bias = nn.Parameter(torch.cat([self.linear_fc.linear.bias, torch.zeros(out_features - self.linear_fc.out_features).cuda()]))
                else:
                    new_linear_fc.linear.weight = self.linear_fc.linear.weight
                    new_linear_fc.linear.bias = self.linear_fc.linear.bias
            del self.linear_fc
            self.linear_fc = new_linear_fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x)
        return out
    
class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):

        return {'logits': F.linear(input, self.weight, self.bias)}


class UnifiedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(UnifiedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # 使用全连接层的前向传播
        logits = self.linear(x)
        # 返回与原型分类器一致的字典格式
        return {'logits': logits}
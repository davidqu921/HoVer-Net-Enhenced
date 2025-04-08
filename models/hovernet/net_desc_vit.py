import math
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

from .net_utils import DenseBlock, Net, TFSamepaddingLayer, UpSample2x
from .utils import crop_op

class HoverIT(Net):
    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        '''
        # 1. 定义 Swin Transformer 的配置参数
        swin_config = {
            "img_size": 256,          # 输入图像尺寸（需与您的数据一致）
            "patch_size": 4,          # Patch大小（4x4像素）
            "in_chans": input_ch,     # 输入通道数（默认为3，医学图像可能需要调整）
            "embed_dim": 128,         # 初始嵌入维度（swin-base为128，swin-tiny为96）
            "depths": [2, 2, 18, 2],  # 各阶段的Transformer层数
            "num_heads": [4, 8, 16, 32],  # 各阶段的多头注意力头数
            "window_size": 7,         # 局部窗口大小（必须是7的倍数）
            "drop_path_rate": 0.2,    # 随机深度衰减率（防止过拟合）
            "pretrained": False       # 这里先设为False，稍后手动加载权重
        }

        # 2. 初始化Swin模型结构
        self.swin = SwinTransformer(**swin_config)  # **表示解包字典参数

        # 3. 加载预训练权重（如果pretrained=True）
        if True:
            from timm.models import swin_base_patch4_window7_224
            pretrained_model = swin_base_patch4_window7_224(pretrained=True)
            
            # 过滤掉分类头权重（因为分类头尺寸可能与任务不匹配）
            state_dict = {
                k: v for k, v in pretrained_model.state_dict().items() 
                if "head" not in k
            }
            
            # 加载权重（strict=False允许部分不匹配）
            self.swin.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained Swin weights (excluding head)")

        '''

        # 初始化 Swin Transformer 编码器
        self.swin = SwinTransformer(
            img_size=256,  # 假设输入为256x256
            patch_size=4,  # 匹配原HoVerNet第一层下采样
            in_chans=input_ch,
            embed_dim=96,
            depths=[2, 2, 6, 2],  # 4 stages对应d0-d3
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
        )

        # 根据mode决定是否添加显式padding
        if mode == 'fast':
            self.pad = TFSamepaddingLayer(ksize=7, stride=1)
        else:
            self.pad = nn.Identity()  # 无操作

        # 保持原HoVerNet的conv_bot
        self.conv_bot = nn.Conv2d(768, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder

        # 根据mode选择解码器的ksize
        ksize = 5 if mode == 'original' else 3
        self.decoder = nn.ModuleDict(
            OrderedDict([
                ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)) if nr_types else None,
            ])
        )
        
        self.upsample2x = UpSample2x()
        self.weights_init()  # 在初始化最后调用


    def forward(self, imgs):
        imgs = imgs / 255.0
        
        # Swin Encoder (兼容timm最新版)
        x = self.swin.patch_embed(imgs)  # [B, C, H, W] -> [B, L, C]
        
        # 处理位置编码（不同timm版本兼容）
        if hasattr(self.swin, 'absolute_pos_embed'):  # 旧版本
            x = x + self.swin.absolute_pos_embed
        elif hasattr(self.swin, 'pos_embed'):  # 新版本
            if self.swin.pos_embed is not None:
                x = x + self.swin.pos_embed
        
        x = self.swin.pos_drop(x)
        
        # 获取多尺度特征 [d0, d1, d2, d3]
        d = []
        for i, layer in enumerate(self.swin.layers):
            x = layer(x)
            if i < len(self.swin.layers) - 1:
                B, L, C = x.shape
                H = W = int(math.sqrt(L))
                feat = x.permute(0, 2, 1).reshape(B, C, H, W)
                d.append(feat)
        
        # 最后一层处理
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        d3 = x.permute(0, 2, 1).reshape(B, C, H, W)
        d3 = self.conv_bot(d3)
        d.append(d3)

        # 根据mode裁剪特征图
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        # 保持原解码器逻辑
        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)
            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)
            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)
            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict
    
    ####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return HoverIT(mode=mode, **kwargs)

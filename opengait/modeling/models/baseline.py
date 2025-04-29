import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from einops import rearrange

class Baseline(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        # 判断主干网络类型
        block_type = model_cfg['backbone_cfg']['block']
        is_3d_backbone = block_type in ['BasicBlock3D', 'BasicBlockP3D']

        self.Backbone = SetBlockWrapper(self.Backbone, is_3d=is_3d_backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs  # 输入数据，标签，_,_,序列长度

        sils = ipts[0]  # sils是步态处理中的轮廓序列。
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts  # 删除输入数据以释放内存
        outs = self.Backbone(sils)  # [n, c, s, h, w]
        
        # 添加补丁，针对不固定帧数能够实现降维
        T_out = outs.shape[2]
        # ✅ 防止训练阶段为 None 时出错
        if seqL is not None:
            if isinstance(seqL, torch.Tensor):
                seqL = torch.clamp(seqL, max=T_out)
            elif isinstance(seqL, list):
                seqL = [min(s, T_out) for s in seqL]
                seqL = torch.tensor([seqL], device=outs.device)

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
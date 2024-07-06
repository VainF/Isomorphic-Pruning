from fastcore.basics import patch_to # for monkey patching
import timm
import torch.nn.functional as F

def patch_timm_forward():
    
    @patch_to(timm.models.vision_transformer.Attention)
    def forward(self, x):
        """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # original implementation: x = x.transpose(1, 2).reshape(B, N, C), this force output dim to be C
        x = x.transpose(1, 2).reshape(B, N, -1) 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
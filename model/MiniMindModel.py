from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Union
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-05):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * x
    
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

def precompute_freqs_cis(dim: int, end: int = 32*1024, rope_base=None, rope_scaling: Optional[dict] = None):
    # 初始化RoPE频率
    freqs, attn_factor = (1.0/(rope_base ** (torch.arange(0, dim, 2)[:(dim//2)].float()/dim)), 1.0)

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow =(
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
        )

    # 推断的长度大于训练长度， 使用缩放
    if rope_scaling is not None and end > orig_max:
        # 波长b到i的映射
        inv_dim = lambda b: (dim * math.log(orig_max/ (2 * math.pi * b))/(2 * math.log(rope_base)))

        # 划分高低纬度
        # low：不需要缩放的高频部分
        # high：需要缩放的低频部分
        low, high = (max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1))
        
        # 计算缩放因子
        # low之前， rapm为0， 在high之后， ramp为1； 在low我和high之间，先行过度
        ramp = torch.clamp(
            (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
            0, 
            1
        )

        # 当 ramp=0时（高频）：系数为1，保持原频率不变
        # 当 ramp=1时（低频）：系数为 factor，频率缩放到原来的1/factor
        # 当 0<ramp<1时（过渡区）：平滑过渡
        freqs = freqs*(1 - ramp + ramp / factor)

    # 根据end，生成位置索引t
    t = torch.arange(end, device=freqs.device).float()

    # 计算外积， 将t和频率相乘，得到每个位置的旋转角度
    freqs = torch.outer(t, freqs).float()

    freqs_cos = (
        torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    )
    freqs_sin = (
        torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    )

    return freqs_cos, freqs_sin

# 边写RoPE
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # [a,b]->[-b,a]
    def rotate_half(x):
        # x.shape[-1]取最后一个维度的重点
        # x[..., :x.shape[-1]//2]取前半部分， x[..., x.shape[-1]//2:]取后半部分
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

    # x_rotated=x*cos+rotate_half(x)*sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))

    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))

    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
       return x
    return (
    x[:, :, :, None, :]
    .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
    .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args:MiniMindConfig):
        super().__init__()

        self.num_key_value_heads=args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads

        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj =nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj =nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj =nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        self.attn_dropout = nn.Dropout(args.attention_dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention  

    
    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    # 投影， 计算q,k,v
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    # 把输入拆分成多个头，用view
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
    # q和k，使用roPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
    # 对于k和v，使用repeat（注意kv cache）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq,xk,xv=(
            xq.transpose(1,2),
            # [bas, n_local_heads, seq_len, head_dim]
            repeat_kv(xk,self.n_rep).transpose(1,2),
            repeat_kv(xv,self.n_rep).transpose(1,2),
        )
    # 进行attention计算， q@k^T / sqrt(d)
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz,1,1,-1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask, self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device), 
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
    # 最后拼接头，输出投影，返回
            if attention_mask is not None:
                extenanded_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extenanded_attention_mask = (1.0 - extenanded_attention_mask) * -1e9
                scores = scores + extenanded_attention_mask
            
            scores = F.softmax(scores, dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)
        # [bsz, n_local_heads, seq_len, head_dim]
        output = output.transpose(1,2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.out_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, args:MiniMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64*((intermediate_size+64-1)//64)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x: torch.Tensor):
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            )

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id:int, config:MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)
       
    def forward(self,hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self,config:MiniMindConfig):
        super().__init__() 
        self.config = config
        self.vocab_size, self.num_hidden_layers=(
            config.vocab_size,
            config.num_hidden_layers
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MiniMindBlock(layer_id=i, config=config) for i in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #RoPE频率预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim = config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling,
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids:Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None
        
        past_key_values = past_key_values or [None] * len(self.layers)

        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_enmbeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents = []

        for layer_idx,(layer,apst_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_enmbeddings,
                past_key_value=apst_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )

            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents

class MiniMindForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config:MiniMindConfig):
        self.config = config
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 权重共享
        # 输出层的权重和输入嵌入层的权重共享
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep:Union[int,torch.Tensor] = 0,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        # logits to keep是整数，那就保留最后n个位置
        # 生成的时候只需要最后的logits来预测下一个token
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        aux_loss = None
        if labels is not None:
            # Shift: 用前n-1个位置预测后n-1个位置
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 展平后计算交叉熵，ignore_index=-100 跳过PAD位置
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            aux_loss=aux_loss,
        )
import torch
import open_clip

try:
    from sgm.modules import GeneralConditioner as CLIP_SDXL
    from sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder2
    from modules.sd_hijack_open_clip import FrozenOpenCLIPEmbedder2WithCustomWords
except:
    print(f"[XL Vec] failed to load `sgm.modules`")
    raise



def get_pooled(clip: CLIP_SDXL, text: str, layer='last', index=-1):
    # cf. sgm/modules/encoders/modules.py:FrozenOpenCLIPEmbedder2
    
    mod = clip.embedders[1]
    if isinstance(mod, FrozenOpenCLIPEmbedder2WithCustomWords):
        mod = mod.wrapped
    
    assert isinstance(mod, FrozenOpenCLIPEmbedder2)
    
    tokens = open_clip.tokenize([text]).to(mod.device)
    
    x = mod.model.token_embedding(tokens)  # [batch_size, n_ctx, d_model]
    x = x + mod.model.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = mod.text_transformer_forward(x, attn_mask=mod.model.attn_mask)

    o = x[layer]
    o = mod.model.ln_final(o)
    
    eot = tokens.argmax(dim=-1)
    p = torch.zeros_like(eot)
    if 0 <= index:
        p[0] = index
    else:
        p[0] = eot.item() + index + 1
    
    real_index = p.item()
    assert 0 <= real_index < 77, f'index={index}, real_index={real_index}'
        
    pooled = (
        o[torch.arange(o.shape[0]), p]
        @ mod.model.text_projection
    )
    
    return pooled, real_index

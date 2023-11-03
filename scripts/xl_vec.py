NAME = 'XL Vec'

from torch import Tensor, FloatTensor, nn
import gradio as gr
from modules.processing import StableDiffusionProcessing
from modules import scripts

from scripts.sdhook import SDHook
from scripts.xl_clip import CLIP_SDXL, get_pooled
from scripts.xl_vec_xyz import init_xyz



def hook_input(
    args: 'Hook',
    mod: nn.Module,
    inputs: tuple[dict[str,Tensor]]
):
    if not args.enabled:
        return
    
    assert isinstance(mod, CLIP_SDXL)
    input = inputs[0]
    overwritten_keys = set()
    
    def create(v: list[float], src: FloatTensor):
        return FloatTensor(v).to(dtype=src.dtype, device=src.device)
    
    def put(name: str, v: list[float]):
        if name in input:
            src = input[name]
            input[name] = create(v, src).reshape(src.shape)
            overwritten_keys.add(name)
    
    old = {k: v for k, v in input.items()}
    
    put('original_size_as_tuple', [args.original_height, args.original_width])
    put('crop_coords_top_left', [args.crop_top, args.crop_left])
    put('target_size_as_tuple', [args.target_height, args.target_width])
    if input['aesthetic_score'].item() == 6.0:
        # positive prompt
        put('aesthetic_score', [args.aesthetic_score])
    else:
        # negative prompt
        put('aesthetic_score', [args.negative_aesthetic_score])
    
    new = {k: v for k, v in input.items()}
    
    for k in overwritten_keys:
        print(f"{k}: {old[k].tolist()} -> {new[k].tolist()}")
    
    return inputs

def hook_output(
    args: 'Hook',
    mod: nn.Module,
    inputs: tuple[dict[str,Tensor]],
    output: dict,
):
    if not args.enabled:
        return
    
    if inputs[0]['aesthetic_score'].item() == 6.0:
        # positive prompt
        prompt = args.extra_prompt
        index = args.token_index
    else:
        # negative prompt
        prompt = args.extra_negative_prompt
        index = args.negative_token_index
    
    if prompt is None or len(prompt) == 0:
        if index < 0:
            # default
            return
        # use original prompt
        prompt = inputs[0]['txt'][0]
    
    assert isinstance(mod, CLIP_SDXL)
    
    try:
        args.enabled = False
        pooled, at = get_pooled(mod, prompt, index=index) # (1,1280)
        assert pooled.shape == (1, 1280), f'pooled.shape={pooled.shape}'
    finally:
        args.enabled = True
    
    output['vector'][:, 0:1280] = pooled[:]
    print(f"vector[:, 0:1280]: {inputs[0]['txt']} -> {[prompt]} @ {at}")

    return output



class Hook(SDHook):

    def __init__(
        self,
        enabled: bool,
        crop_left: float,
        crop_top: float,
        original_width: float,
        original_height: float,
        target_width: float,
        target_height: float,
        aesthetic_score: float,
        negative_aesthetic_score: float,
        extra_prompt: str|None,
        extra_negative_prompt: str|None,
        token_index: int|float,
        negative_token_index: int|float,
    ):
        super().__init__(enabled)
        self.crop_left = float(crop_left)
        self.crop_top = float(crop_top)
        self.original_width = float(original_width)
        self.original_height = float(original_height)
        self.target_width = float(target_width)
        self.target_height = float(target_height)
        self.aesthetic_score = float(aesthetic_score)
        self.negative_aesthetic_score = float(negative_aesthetic_score)
        self.extra_prompt = extra_prompt
        self.extra_negative_prompt = extra_negative_prompt
        self.token_index = int(token_index)
        self.negative_token_index = int(negative_token_index)
    
    def hook_clip(self, p: StableDiffusionProcessing, clip: nn.Module):
        if not hasattr(p.sd_model, 'is_sdxl') or not p.sd_model.is_sdxl:
            return
        
        def inp(*args, **kwargs):
            return hook_input(self, *args, **kwargs)
        
        def outp(*args, **kwargs):
            return hook_output(self, *args, **kwargs)
        
        self.hook_layer_pre(clip, inp)
        self.hook_layer(clip, outp)



class Script(scripts.Script):
    
    def __init__(self):
        super().__init__()
        self.last_hooker: SDHook|None = None
    
    def title(self):
        return NAME
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        with gr.Accordion(NAME, open=False):
            enabled = gr.Checkbox(label='Enabled', value=False)
            crop_left = gr.Slider(minimum=-512, maximum=512, step=1, value=0, label='Crop Left')
            crop_top = gr.Slider(minimum=-512, maximum=512, step=1, value=0, label='Crop Top')
            original_width = gr.Slider(minimum=0, maximum=4096, step=1, value=0, label='Original Width')
            original_height = gr.Slider(minimum=0, maximum=4096, step=1, value=0, label='Original Height')
            target_width = gr.Slider(minimum=0, maximum=4096, step=1, value=0, label='Target Width')
            target_height = gr.Slider(minimum=0, maximum=4096, step=1, value=0, label='Target Height')
            aesthetic_score = gr.Slider(minimum=0.0, maximum=10.0, step=0.05, value=6.0, label="Aesthetic Score (0..10)")
            negative_aesthetic_score = gr.Slider(minimum=0.0, maximum=10.0, step=0.05, value=2.5, label="Negative Aesthetic Score (0..10)")
            extra_prompt = gr.Textbox(lines=3, label='Extra prompt (set empty to be disabled)')
            extra_negative_prompt = gr.Textbox(lines=3, label='Extra negative prompt (set empty to be disabled)')
            token_index = gr.Slider(minimum=-77, maximum=76, step=1, value=-1, label='Token index in the prompt for the vector (-1 is first EOT)')
            negative_token_index = gr.Slider(minimum=-77, maximum=76, step=1, value=-1, label='Token index in the negative prompt for the vector (-1 is first EOT)')
        return [
            enabled,
            crop_left,
            crop_top,
            original_width,
            original_height,
            target_width,
            target_height,
            aesthetic_score,
            negative_aesthetic_score,
            extra_prompt,
            extra_negative_prompt,
            token_index,
            negative_token_index,
        ]
    
    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        crop_left: float,
        crop_top: float,
        original_width: float,
        original_height: float,
        target_width: float,
        target_height: float,
        aesthetic_score: float,
        negative_aesthetic_score: float,
        extra_prompt: str,
        extra_negative_prompt: str,
        token_index: float,
        negative_token_index: float,
    ):
        
        if self.last_hooker is not None:
            self.last_hooker.__exit__(None, None, None)
            self.last_hooker = None
        
        if not enabled:
            return
        
        self.last_hooker = Hook(
            enabled=True,
            crop_left=crop_left,
            crop_top=crop_top,
            original_width=original_width,
            original_height=original_height,
            target_width=target_width,
            target_height=target_height,
            aesthetic_score=aesthetic_score,
            negative_aesthetic_score=negative_aesthetic_score,
            extra_prompt=extra_prompt,
            extra_negative_prompt=extra_negative_prompt,
            token_index=token_index,
            negative_token_index=negative_token_index,
        )

        self.last_hooker.setup(p)
        self.last_hooker.__enter__()
        
        p.extra_generation_params.update({
            f'[{NAME}] Enabled': enabled,
            f'[{NAME}] Crop Left': crop_left,
            f'[{NAME}] Crop Top': crop_top,
            f'[{NAME}] Original Width': original_width,
            f'[{NAME}] Original Height': original_height,
            f'[{NAME}] Target Width': target_width,
            f'[{NAME}] Target Height': target_height,
            f'[{NAME}] Aesthetic Score': aesthetic_score,
            f'[{NAME}] Negative Aesthetic Score': negative_aesthetic_score,
            f'[{NAME}] Extra Prompt': extra_prompt.__repr__(),
            f'[{NAME}] Extra Negative Prompt': extra_negative_prompt.__repr__(),
            f'[{NAME}] Token Index': token_index,
            f'[{NAME}] Negative Token Index': negative_token_index,
        })

        if hasattr(p, 'cached_c'):
            p.cached_c = [None, None]
        if hasattr(p, 'cached_uc'):
            p.cached_uc = [None, None]



init_xyz(Script, NAME)

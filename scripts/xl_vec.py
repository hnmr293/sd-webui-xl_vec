NAME = 'XL Vec'

from torch import Tensor, FloatTensor, nn
import gradio as gr
from modules.processing import StableDiffusionProcessing
from modules import scripts

from scripts.sdhook import SDHook
from scripts.xl_vec_xyz import init_xyz

try:
    from sgm.modules import GeneralConditioner as CLIP_SDXL
except:
    print(f"[{NAME}] failed to load `sgm.modules.GeneralConditioner`")
    raise



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

    
    def hook_clip(self, p: StableDiffusionProcessing, clip: nn.Module):
        
        if not hasattr(p.sd_model, 'is_sdxl') or not p.sd_model.is_sdxl:
            return
        
        def hook(mod: nn.Module, inputs: tuple[dict[str,Tensor]]):
            if not self.enabled:
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
            
            put('original_size_as_tuple', [self.original_height, self.original_width])
            put('crop_coords_top_left', [self.crop_top, self.crop_left])
            put('target_size_as_tuple', [self.target_height, self.target_width])
            if input['aesthetic_score'].item() == 6.0:
                # positive prompt
                put('aesthetic_score', [self.aesthetic_score])
            else:
                # negative prompt
                put('aesthetic_score', [self.negative_aesthetic_score])
            
            new = {k: v for k, v in input.items()}
            
            for k in overwritten_keys:
                print(f"{k}: {old[k].tolist()} -> {new[k].tolist()}")
            
            return inputs
        
        self.hook_layer_pre(clip, hook)


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
        })

        if hasattr(p, 'cached_c'):
            p.cached_c = [None, None]
        if hasattr(p, 'cached_uc'):
            p.cached_uc = [None, None]



init_xyz(Script, NAME)

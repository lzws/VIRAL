from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import snapshot_download
from PIL import Image
import torch
import pandas as pd
# Load models
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
lora_path = " "
lora = load_state_dict(lora_path, torch_dtype=torch.bfloat16, device=device)
moelora_target_modules = ["img_mlp.net.2"]
lora_base = {k:v for k,v in lora.items() if not any(m in k for m in moelora_target_modules)}
lora_moe = {k:v for k,v in lora.items() if any(m in k for m in moelora_target_modules)}

pipe.load_lora(pipe.dit, state_dict=lora_base)

model_ = replace_target_modules_with_moe_lora(
    getattr(pipe, 'dit'),
    target_modules=moelora_target_modules,
    num_experts=4,r=64,lora_alpha=64,lora_dropout=-1,top_k=2,upcast_dtype=pipe.torch_dtype,
    moe_type='tokenmoe'
)
load_moe_res = model_.load_state_dict(lora_moe,strict=False)
if len(load_moe_res[1]) > 0:
        print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_moe_res[1]}")
setattr(pipe, 'dit', model_.to(pipe.device,pipe.torch_dtype))



datas = pd.read_csv("test.csv")

for i,row in datas.iterrows():
    if i<10:
        continue
    img1 = row["img1"]
    tgt1 = row["tgt1"]
    img2 = row["img2"]
    edit_image = [
        Image.open(img1).convert("RGB"),
        Image.open(tgt1).convert("RGB"),
        Image.open(img2).convert("RGB")
    ]
    prompt = " "
    negative_prompt = "  "


    image = pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        edit_image=edit_image,
        seed=1,
        num_inference_steps=40,
        zero_cond_t=True,
    )
    image.save(f"test/{i}_tgt2.png")
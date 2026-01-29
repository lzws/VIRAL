import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset,IncontextDataset
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.core.data.operators import *
from diffsynth.core import load_state_dict
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        lora_base_model_2=None, lora_target_modules_2="",
        moelora_base_model=None,moelora_target_modules="",
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        zero_cond_t=False,
        if_prompt=False,
        num_experts=4,moe_lora_rank=32, moe_lora_alpha=16, moe_lora_dropout=0.1, top_k=1, moe_type="moe",
        auto_resize=False

    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, processor_config=processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
            moelora_base_model=moelora_base_model,moelora_target_modules=moelora_target_modules,
            num_experts=num_experts,moe_lora_rank=moe_lora_rank, moe_lora_alpha=moe_lora_alpha, moe_lora_dropout=moe_lora_dropout, top_k=top_k, moe_type=moe_type
        )

        if lora_base_model_2 is not None:
            model2 = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model_2),
                target_modules=lora_target_modules_2.split(","),
                lora_rank=32,
                lora_alpha=32
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                moelora_target_modules_ = moelora_target_modules.split(",")
                state_dict = {k: v for k, v in state_dict.items() if not any(m in k for m in moelora_target_modules_)}
                state_dict = {k.replace('text_encoder.', ''): v for k, v in state_dict.items() if 'text_encoder' in k}
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model2.load_state_dict(state_dict, strict=False)
                print(f"lora_base_nodel_2 LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(self.pipe, lora_base_model_2, model2)
            print('Add lora to model2')
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.zero_cond_t = zero_cond_t
        self.auto_resize = auto_resize
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }

        self.if_prompt=if_prompt
        self.moe_type=moe_type
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": self.auto_resize,
            "zero_cond_t": self.zero_cond_t,
            "edit_nums":3,
            "if_prompt":self.if_prompt
        }
        if self.auto_resize:
            print("Auto resize is enabled.")
        # Assume you are using this pipeline for inference,
        # please fill in the input parameters.
        if isinstance(data["image"], list):
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"][0].size[1],
                "width": data["image"][0].size[0],
            })
        else:
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"].size[1],
                "width": data["image"].size[0],
            })
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        if self.moe_type == "dismoe" or self.moe_type == "tokenmoe":

            all_aux_losses = []
            for module in model.modules():
                if hasattr(module, "current_aux_loss"):
                    all_aux_losses.append(module.current_aux_loss)


            if len(all_aux_losses) > 0:

                avg_aux_loss = torch.stack(all_aux_losses).mean()
            else:
                avg_aux_loss = 0.0
            if len(all_aux_losses) > 0:
                print(f"loss: {loss.item()}, aux_loss: {avg_aux_loss.item()}")
            else:
                print(f"loss: {loss.item()}")
            return loss + 0.005 * avg_aux_loss
        else:
            print(f"loss: {loss.item()}")
            return loss


def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--zero_cond_t", default=False, action="store_true", help="A special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.")
    parser.add_argument("--train_type", type=str, default='no', help="")
    parser.add_argument("--incontext_metadata_path", type=str, default=0.01, help="")
    parser.add_argument("--style_metadata_path", type=str, default=None, help="")
    parser.add_argument("--openpose_metadata_path", type=str, default=None, help="")
    parser.add_argument("--segmentation_metadata_path", type=str, default=None, help="")
    parser.add_argument("--detection_metadata_path", type=str, default=None, help="")
    parser.add_argument("--extension_metadata_path", type=str, default=None, help="")
    parser.add_argument("--select_metadata_path", type=str, default=None, help="")
    parser.add_argument("--incontext2_metadata_path", type=str, default=None, help="")
    parser.add_argument("--derain_metadata_path", type=str, default=None, help="")
    parser.add_argument("--enhance_metadata_path", type=str, default=None, help="")
    parser.add_argument("--gopro_metadata_path", type=str, default=None, help="")
    parser.add_argument("--generate_metadata_path", type=str, default=None, help="")
    parser.add_argument("--general_metadata_path", type=str, default=None, help="")
    parser.add_argument("--general2_metadata_path", type=str, default=None, help="")
    parser.add_argument("--omniedit_metadata_path", type=str, default=None, help="omniedit_metadata_path")
    parser.add_argument("--imgcluster_metadata_path", type=str, default=None, help="--imgcluster_metadata_path")
    parser.add_argument("--lora_base_model_2", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--learning_rate_2", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--lora_target_modules_2", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--moelora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--moelora_target_modules", type=str, default="txt_mlp.net.2,img_mlp.net.2", help="which layers moe LoRA is added to")
    parser.add_argument("--if_prompt", default=False, action="store_true", help="if use system prompt")
    parser.add_argument("--moe_lora_dropout", type=float, default=0.1, help="moe dropout")
    parser.add_argument("--num_experts", type=int, default=1, help="moe experts nums")
    parser.add_argument("--moe_lora_rank", type=int, default=32, help="moe lora rank")
    parser.add_argument("--moe_lora_alpha", type=int, default=32, help="moe lora alpha")
    parser.add_argument("--top_k", type=int, default=1, help="moe tok-k experts")
    parser.add_argument("--moe_type", type=str, default="moe", help="moe type")
    parser.add_argument("--auto_resize", default=False, action="store_true", help="if use system prompt")
    return parser


if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()
    print("##"*25)
    print(f"训练用的参数：{args}")
    print("##"*25)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    if args.train_type == 'incontext':
        dataset = IncontextDataset(args=args)
    else:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_image_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
            ),
            special_operator_map={
                # Qwen-Image-Layered
                "layer_input_image": ToAbsolutePath(args.dataset_base_path) >> LoadImage(convert_RGB=False, convert_RGBA=True) >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
                "image": RouteByType(operator_map=[
                    (str, ToAbsolutePath(args.dataset_base_path) >> LoadImage() >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16)),
                    (list, SequencialProcess(ToAbsolutePath(args.dataset_base_path) >> LoadImage(convert_RGB=False, convert_RGBA=True) >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16))),
                ])
            }
        )
    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_base_model_2=args.lora_base_model_2,
        lora_target_modules_2=args.lora_target_modules_2,
        moelora_base_model=args.moelora_base_model,
        moelora_target_modules=args.moelora_target_modules,
        if_prompt=args.if_prompt,
        num_experts=args.num_experts,moe_lora_rank=args.moe_lora_rank, moe_lora_alpha=args.moe_lora_alpha, moe_lora_dropout=args.moe_lora_dropout, top_k=args.top_k, moe_type=args.moe_type,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=accelerator.device,
        zero_cond_t=args.zero_cond_t,
        auto_resize=args.auto_resize
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)

accelerate launch examples/qwen_image/model_training/train.py \
  --train_type "incontext" \
  --incontext_metadata_path newdataset/train_set/incontext.csv \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_qwen_imgae_edit_multi.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --remove_prefix_in_ckpt "pipe." \
  --lora_base_model "dit" \
  --output_path "./models/train/1218/Qwen-Image-Edit-2511_lora_tokenmoe_512_rank128_top1_gradacc4" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mod.1,txt_mod.1,txt_mlp.net.2" \
  --lora_rank 128 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --zero_cond_t \
  --moelora_base_model "dit" \
  --moelora_target_modules "img_mlp.net.2" \
  --num_experts 4 \
  --moe_lora_rank 128 \
  --moe_lora_alpha 128 \
  --top_k 1 \
  --moe_type "tokenmoe" \



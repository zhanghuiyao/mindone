python examples/rec_video_folder.py \
    --batch_size 1 \
    --real_video_dir datasets/UCF-101/ \
    --data_file_path datasets/ucf101_test.csv \
    --generated_video_dir recons/ucf101_test/ \
    --device Ascend \
    --sample_fps 30 \
    --sample_rate 1 \
    --num_frames 25 \
    --height 256 \
    --width 256 \
    --num_workers 8 \
    --ae "WFVAEModel_D8_4x8x8" \
    --ae_path LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    # --ms_checkpoint path/to/ms/ckpt

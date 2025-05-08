#!/bin/bash
job_name='v1011-MLRG-ft-RCB'
python ../main_v0926_ablation_study.py \
--data_name mimic_cxr \
--version "${job_name}" \
--task "finetune" \
--ann_path "/home/20031211471/Code/Code/Data/five_work_mimic_cxr_annotation_v1.1.json" \
--view_position_embed "/home/20031211471/Code/Code/Data/five_work_mimic_cxr_view_position2idx.json" \
--images_dir "/home/20031211471/Data/MIMIC-CXR/files/" \
--max_length 100 \
--is_save_checkpoint "yes" \
--is_multiview_learning "yes" \
--is_prior_scan "yes" \
--using_mpc_loss "no" \
--is_indication "yes" \
--is_prior_report "yes" \
--ckpt_zoo_dir "/home/20031211471/Data/checkpoints/" \
--load "/home/20031211471/Code/Code/five_gpt2/script/results/mimic_cxr/pretrain/v0906_fs_pt_2024_09_06_12/checkpoint/best_model.ckpt" \
--cvt2distilgpt2_path "mimic_cxr_jpg_chen/cvt_21_to_gpt2/epoch=8-val_chen_cider=0.425092.ckpt" \
--pt_lr 5.0e-6 \
--ft_lr 5.0e-5 \
--monitor_metric "RCB" \
--epochs 50 \
--num_workers 8 \
--batch_size 14

#!/bin/bash
job_name='v1011-MLRG-ft-RCB'
python ../main_v0926_ablation_study.py \
--data_name mimic_cxr \
--version "${job_name}" \
--task "finetune" \
--ann_path "https://huggingface.co/MK-runner/MLRG/blob/main/radiology%20report/five_work_mimic_cxr_annotation_v1.1.json" \
--view_position_embed "https://huggingface.co/MK-runner/MLRG/blob/main/radiology%20report/five_work_mimic_cxr_view_position_v1.1.json" \
--images_dir "Data/MIMIC-CXR/files/" \
--max_length 100 \
--is_save_checkpoint "yes" \
--is_multiview_learning "yes" \
--is_prior_scan "yes" \
--using_mpc_loss "no" \
--is_indication "yes" \
--is_prior_report "yes" \
--ckpt_zoo_dir "Data/checkpoints/" \
--load "https://huggingface.co/MK-runner/MLRG/blob/main/mimic-cxr/pretrain/best_model.ckpt" \
--cvt2distilgpt2_path "mimic_cxr_jpg_chen/cvt_21_to_gpt2/epoch=8-val_chen_cider=0.425092.ckpt" \
--pt_lr 5.0e-6 \
--ft_lr 5.0e-5 \
--monitor_metric "RCB" \
--epochs 50 \
--num_workers 8 \
--batch_size 14

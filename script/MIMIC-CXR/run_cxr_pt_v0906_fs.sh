#!/bin/bash
job_name='v0906_fs_pt'
python ../main_v0926_ablation_study.py \
--data_name mimic_cxr \
--version "${job_name}" \
--task "pretrain" \
--ann_path "https://huggingface.co/MK-runner/MLRG/blob/main/radiology%20report/five_work_mimic_cxr_annotation_v1.1.json" \
--view_position_embed "https://huggingface.co/MK-runner/MLRG/blob/main/radiology%20report/five_work_mimic_cxr_view_position_v1.1.json" \
--images_dir "Data/MIMIC-CXR/files/" \
--max_length 100 \
--is_save_checkpoint "yes" \
--is_multiview_learning "yes" \
--is_indication "yes" \
--ckpt_zoo_dir "Data/checkpoints/" \
--report_style "factual_serialization" \
--pt_lr 5.0e-5 \
--epochs 50 \
--batch_size 32

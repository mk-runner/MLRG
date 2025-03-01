import argparse
import datetime
import random
import threading
import os
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from dateutil import tz
from typing import Tuple
import matplotlib.pyplot as plt
import torch
from dateutil import tz
from torch import Tensor, device


def mimic_cxr_image_path(image_dir, subject_id, study_id, dicom_id, ext='dcm'):
    return os.path.join(image_dir, 'p' + str(subject_id)[:2], 'p' + str(subject_id),
                        's' + str(study_id), str(dicom_id) + '.' + ext)


def mimic_cxr_text_path(image_dir, subject_id, study_id, ext='txt'):
    return os.path.join(image_dir, 'p' + str(subject_id)[:2], 'p' + str(subject_id),
                        's' + str(study_id) + '.' + ext)

def enumerated_save_path(save_dir, save_name, extension):
    save_path = os.path.join(save_dir, save_name + extension)
    assert '.' in extension, 'No period in extension.'
    if os.path.isfile(save_path):
        count = 2
        while True:
            save_path = os.path.join(save_dir, save_name + "_" + str(count) + extension)
            count += 1
            if not os.path.isfile(save_path):
                break

    return save_path


def str2bool(value):
    if value.lower() in ['yes', 'true', 't', '1']:
        return True
    elif value.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_arguments():
    # -------------------------------
    # load hyper-param
    # -------------------------------
    parse = argparse.ArgumentParser()
    # basic configuration
    parse.add_argument('--task', type=str, default='test',
                       choices=['pretrain', 'finetune', 'test'])
    # data configuration
    parse.add_argument('--data_name', type=str,
                       choices=['mimic_cxr', 'mimic_abn', 'twoview_cxr'], default='mimic_cxr')
    parse.add_argument('--ann_path', type=str, help='annotation for radiology reports',
                       default='/home/miao/data/dataset/MIMIC-CXR/five_work_mimic_cxr_annotation_v1.1.json',
                       # default='/home/miao/data/dataset/MIMIC-CXR/MIMIC-ABN/mlrg_mimic_abn_annotation_v1.1.json',
                       # default='/home/miao/data/dataset/MIMIC-CXR/mlrg_multiview_cxr_annotation_v1.1.json',
                       )
    parse.add_argument('--view_position_embed', type=str, help='the local path of viewposition2id',
                       default='/home/miao/data/dataset/MIMIC-CXR/five_work_mimic_cxr_view_position_v1.1.json'
                       )
    parse.add_argument('--images_dir', type=str,
                       default='/home/miao/data/dataset/MIMIC-CXR/files/'
                       )
    parse.add_argument('--max_length', type=int, default=100)
    parse.add_argument('--num_workers', type=int, default=8)
    parse.add_argument('--is_save_checkpoint', type=str2bool, default='yes', help='whether save checkpoint')
    # parse.add_argument('--online_checkpoint', action='store_true', help='whether using online checkpoint')
    parse.add_argument('--is_multiview_learning', type=str2bool, default='yes', help='whether using multiple positive contrastive learning')
    parse.add_argument('--is_prior_scan', type=str2bool, default='yes', help='whether using prior scan for pretraining')
    parse.add_argument('--using_mpc_loss', type=str2bool, default='yes', help='whether multi-positive contrastive loss for pretraining')
    parse.add_argument('--using_local_loss', type=str2bool, default='no', help='whether token-wise cross-modal alignment loss')
    parse.add_argument('--is_prior_report', type=str2bool, default='yes', help='whether using prior report for finetune')
    parse.add_argument('--is_indication', type=str2bool, default='yes', help='whether using indication')
    # parse.add_argument('--is_wandb', type=str2bool, default='no', help='whether using wandb to record results.')
    parse.add_argument('--report_style', type=str, choices=['report', 'factual_serialization'],
                       default='factual_serialization', help='the style of reports for cross-modal alignment')
    parse.add_argument('--ckpt_zoo_dir', type=str,
                       default='/home/miao/data/dataset/checkpoints/',
                       help='if using local checkpoint, this variable must be provided')
    parse.add_argument('--text_encoder_num_layers', type=int, default=6)
    parse.add_argument('--cross_modal_fusion_num_layers', type=int, default=1)
    parse.add_argument('--multiview_fusion_num_layers', type=int, default=3)
    parse.add_argument('--num_heads', type=int, default=8)

    # trainer configuration
    parse.add_argument('--pt_lr', type=float, default=5.0e-6)  # 5.0e-5
    parse.add_argument('--ft_lr', type=float, default=5.0e-5)  # 5.0e-5
    parse.add_argument('--temp', type=float, default=0.5,
                       help='temperature parameter for instance-wise cross-modal alignment')  # 5.0e-5
    parse.add_argument('--monitor_metric', type=str, default='RCB',
                       help='the metric is used to selecting best models. pretraining is all_loss, while fine-tuning is RCB')
    # choices={all_metrics, metrics, RC, RB, RCB}
    parse.add_argument('--epochs', type=int, default=50)
    parse.add_argument('--batch_size', type=int, default=2)
    parse.add_argument('--hidden_size', type=int, default=768,
                       help='the dimension of unify embedding for image and text features')
    parse.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.',
                       # default='script/results/mimic_cxr/finetune/v0915_ft-fs_2024_09_17_10/checkpoint/last-1.0329.ckpt'
                       )
    parse.add_argument('--load', type=str, help='whether to load the pre-trained model.',
                       # default='script/results/mimic_cxr/pretrain/v0906_fs_pt_2024_09_06_12/checkpoint/best_model.ckpt'
                       # default='script/results/mimic_cxr/finetune/v0915_ft-fs_2024_09_17_10/checkpoint/last-1.0329.ckpt'
                       )
    parse.add_argument('--test_ckpt_path', type=str, help='checkpoint for test',
                       default='/home/miao/data/Code/five_gpt2/script/results/mimic_cxr/finetune/v1011-MLRG-ft-RCB_2024_10_12_14/best_model.ckpt',
                       )
    parse.add_argument('--version', type=str, default='long_sentence', help='the name of experiment')
    # ========= metrics checkpoint config =====#
    parse.add_argument('--chexbert_path', type=str, default='chexbert.pth', help='checkpoint')
    parse.add_argument('--bert_path', type=str, default='bert-base-uncased', help='checkpoint')
    parse.add_argument('--rad_dino_path', type=str, default='microsoft/rad-dino', help='checkpoint')
    parse.add_argument('--radgraph_path', type=str, default='radgraph', help='checkpoint')
    # ========= backbone checkpoint config =====#
    parse.add_argument('--cxr_bert_path', type=str,
                       default='microsoft/BiomedVLP-CXR-BERT-specialized', help='checkpoint')
    parse.add_argument('--distilgpt2_path', type=str,
                       default='distilbert/distilgpt2',
                       help='text decoder checkpoint')
    parse.add_argument('--cvt2distilgpt2_path', type=str,
                       default='mimic_cxr_jpg_chen/cvt_21_to_gpt2/epoch=8-val_chen_cider=0.425092.ckpt',
                       help='baseline checkpoint')
    # ========= implementation config =====#
    parse.add_argument('--seed', type=int, default=9233, help='random seed')
    parse.add_argument('--num_gpus', type=int, default=1, help='the number of gpus')
    parse.add_argument('--num_beams', type=int, default=3, help='beam size for language generation')
    # ========= record results ============#
    parse.add_argument('--save_period', type=int, default=1, help='the period of saved files.')
    parse.add_argument('--exp_dir_trial', type=str, default='results',
                       help='fold path for recording experimental results')
    parse.add_argument('--print_step', type=int, default=500, help='the frequency of print')
    # =============finish=====================#
    args = parse.parse_args()
    args = vars(args)
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H")
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['exp_dir_trial'] = f'{args["exp_dir_trial"]}/{args["data_name"]}/{args["task"]}/{args["version"]}_{extension}'
    os.makedirs(args['exp_dir_trial'], exist_ok=True)

    # config logger
    logger = SetLogger(f'{args["exp_dir_trial"]}/log_{extension}.log','a')

    # determine using_mpc_loss
    if not args['is_multiview_learning']:
        args['using_mpc_loss'] = False

    # determine absolute path for checkpoints
    # if not args['online_checkpoint']:
    candi_list = ['chexbert_path', 'radgraph_path', "bert_path", "cxr_bert_path",
                  "cvt2distilgpt2_path", "distilgpt2_path", "rad_dino_path"]
    for candi in candi_list:
        if args[candi] is None:
            continue
        args[candi] = os.path.join(args['ckpt_zoo_dir'], args[candi])
    # determine the monitor_mode
    args['monitor_mode'] = 'max'
    if args['task'] == 'pretrain':  # pretrain
        args['monitor_mode'] = 'min'
        args['monitor_metric'] = 'val_epoch_loss'
    # args['unify_dim'] = args['d_vf']
    checkpoint_dir = os.path.join(args['exp_dir_trial'], 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    args['checkpoint_dir'] = checkpoint_dir
    args['time'] = extension
    # save parameters
    config_dir = f"{args['exp_dir_trial']}/configs"
    os.makedirs(config_dir, exist_ok=True)
    file_name = f"{config_dir}/config_{extension}.yaml"
    print(f'parameters is saved in {file_name}')
    with open(file_name, 'w') as file:
        yaml.dump(args, file, default_flow_style=False)

    return args, logger
    

def setup_seed(seed):
    # seed init
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch seed init
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SetLogger:
    def __init__(self, filepath, mode='a', lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi-process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            raise ValueError("Mode must be 'w' or 'a'")
        self.mode = mode
        self.lock = lock or threading.Lock()

        try:
            self.file = open(self.filepath, self.mode)
        except Exception as e:
            print(f"Failed to open log file: {e}")
            raise

    def info(self, message):
        """
        Log an info message to the file.
        :param message: The message to log
        """
        with self.lock:
            try:
                self.file.write(message + '\n')
                self.file.flush()
            except Exception as e:
                print(f"Failed to write to log file: {e}")

    def __del__(self):
        """Ensure that the file is closed when the logger is destroyed."""
        try:
            if not self.file.closed:
                self.file.close()
        except Exception as e:
            print(f"Failed to close log file: {e}")


def get_extended_attention_mask(
        attention_mask: Tensor, input_shape: Tuple[int], device: device = None, dtype: torch.float = None,
        is_decoder: bool = False
) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
        device
        dtype
        is_decoder
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if dtype is None:
        dtype = torch.float

    if device is None:
        device = attention_mask.device

    if not (attention_mask.dim() == 2 and is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            extended_attention_mask = create_extended_attention_mask_for_decoder(
                input_shape, attention_mask, device
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
    if device is not None:
        warnings.warn(
            "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
        )
    else:
        device = attention_mask.device
    batch_size, seq_length = input_shape
    seq_ids = torch.arange(seq_length, device=device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
    # causal and attention masks must have same type with pytorch version < 1.3
    causal_mask = causal_mask.to(attention_mask.dtype)

    if causal_mask.shape[1] < attention_mask.shape[1]:
        prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
        causal_mask = torch.cat(
            [
                torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                causal_mask,
            ],
            axis=-1,
        )

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    return extended_attention_mask


import os
import torch

import pytorch_lightning as pl
from transformers import GPT2TokenizerFast
from pytorch_lightning import seed_everything

from tools.utils import setup_arguments, setup_seed
from models.model_pretrain_finetune_v0905_ablation import Pretrain, Finetune

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_float32_matmul_precision('medium')


def main():
    args, logger = setup_arguments()
    setup_seed(args['seed'])
    seed_everything(args['seed'])
    if args['is_save_checkpoint']:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=False,
            monitor=args['monitor_metric'],
            mode=args['monitor_mode'],
            save_last=True,
            save_weights_only=False,
            dirpath=args['checkpoint_dir'],
            filename='best_model'
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=0,
            verbose=False,
            monitor=args['monitor_metric'],
            mode=args['monitor_mode'],
            save_last=False,
            save_weights_only=False,
        )

    earlystop_callback = pl.callbacks.EarlyStopping(
        monitor=args["monitor_metric"],
        patience=15,
        verbose=False, mode=args['monitor_mode']
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, lr_callback, earlystop_callback]

    tokenizer = GPT2TokenizerFast.from_pretrained(args['distilgpt2_path'])
    tokenizer.add_special_tokens(
        {"bos_token": "[BOS]", 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]',
         'mask_token': '[MASK]', "eos_token": '[EOS]', 'unk_token': '[UNK]'})
    tokenizer.add_tokens(['[NHI]', '[NHPR]'])
    # Print the special tokens:
    print('Description, Special token, Index')
    for k, v in tokenizer.special_tokens_map.items():
        if k != 'additional_special_tokens':
            print(f'{k}, {v}, {getattr(tokenizer, k + "_id")}')
        else:
            for i, j in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids):
                print(f'additional_special_token, {i}, {j}')
    # print the config
    params = ''
    for key, value in args.items():
        params += f'{key}:\t{value}\n'
    logger.info(params)
    # print(params)

    # Trainer
    # Training Hyper-Parameters
    trainer = pl.Trainer(
        accelerator="gpu",
        accumulate_grad_batches=2,
        gradient_clip_val=0.1,
        benchmark=False,
        callbacks=callbacks,
        check_val_every_n_epoch=None,
        strategy="ddp_find_unused_parameters_true",
        devices=args["num_gpus"],
        precision="bf16-mixed",
        deterministic=True,
        max_epochs=args['epochs'],
        logger=None,
        log_every_n_steps=50,
        enable_model_summary=True,
        profiler="simple",

    )
    if args['task'] == 'pretrain':
        model = Pretrain(args, tokenizer, logger)
        if args['load'] is not None:
            cur_model_state = model.state_dict()
            pre_model_state = torch.load(args['load'])['state_dict']
            valid_state = {k: v for k, v in pre_model_state.items() if k in cur_model_state and v.shape == cur_model_state[k].shape}
            cur_model_state.update(valid_state)
            model.load_state_dict(cur_model_state)

        if args['resume'] is not None:
            trainer.fit(model=model, ckpt_path=args['resume'])
        else:
            trainer.fit(model=model)
    elif args['task'] == 'finetune':
        model = Finetune(args, tokenizer, logger)
        if args['resume'] is not None:
            trainer.fit(model=model, ckpt_path=args['resume'])
        else:
            if args['load'] is not None:  # using load checkpoint to initialize
                cur_model_state = model.state_dict()
                pre_model_state = torch.load(args['load'])['state_dict']
                valid_state = {k: v for k, v in pre_model_state.items() if k in cur_model_state and v.shape == cur_model_state[k].shape}
                cur_model_state.update(valid_state)
                model.load_state_dict(cur_model_state)
            trainer.fit(model=model)
    else:     # test
        assert args['test_ckpt_path'] is not None
        model = Finetune(args, tokenizer, logger)
        trainer.test(model=model, ckpt_path=args['test_ckpt_path'])


if __name__ == '__main__':
    main()

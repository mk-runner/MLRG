from typing import List, Union, Any

from tools.metrics.natural_language import NaturalLanguage
from tools.metrics.Radgraph import F1RadGraph
from tools.metrics.f1chexbert import F1CheXbert
from tools.utils import enumerated_save_path
import os
import pandas as pd
import time
import torch

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

CONDITIONS: list[Union[str, Any]] = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]


class CheXbertMetrics(NaturalLanguage):
    is_differentiable = False
    full_state_update = False

    def __init__(
            self,
            bert_path,
            checkpoint_path,
            mbatch_size=16,
            save_class_scores=False,
            save_outputs=False,
            exp_dir=None,
    ):
        super().__init__(dist_sync_on_step=False)
        self.bert_path = bert_path
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.save_class_scores = save_class_scores
        self.save_outputs = save_outputs
        self.exp_dir = exp_dir

    def mini_batch(self, iterable, mbatch_size=1):
        length = len(iterable)
        for i in range(0, length, mbatch_size):
            yield iterable[i:min(i + mbatch_size, length)]

    def compute(self):

        chexbert = CheXbert(
            bert_path=self.bert_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        ).to(self.device)

        table = {'chexbert_y_hat': [], 'chexbert_y': [], 'y_hat': [], 'y': [], 'ids': []}
        for i in self.mini_batch(self.pairs, self.mbatch_size):
            y_hat, y, ids = zip(*i)
            table['chexbert_y_hat'].extend([i + [j] for i, j in zip(chexbert(list(y_hat)).tolist(), list(ids))])
            table['chexbert_y'].extend([i + [j] for i, j in zip(chexbert(list(y)).tolist(), list(ids))])
            table['y_hat'].extend(y_hat)
            table['y'].extend(y)
            table['ids'].extend(ids)

        if torch.distributed.is_initialized():  # If DDP

            chexbert_y_hat_gathered = [None] * torch.distributed.get_world_size()
            chexbert_y_gathered = [None] * torch.distributed.get_world_size()
            y_hat_gathered = [None] * torch.distributed.get_world_size()
            y_gathered = [None] * torch.distributed.get_world_size()
            ids_gathered = [None] * torch.distributed.get_world_size()

            torch.distributed.all_gather_object(chexbert_y_hat_gathered, table['chexbert_y_hat'])
            torch.distributed.all_gather_object(chexbert_y_gathered, table['chexbert_y'])
            torch.distributed.all_gather_object(y_hat_gathered, table['y_hat'])
            torch.distributed.all_gather_object(y_gathered, table['y'])
            torch.distributed.all_gather_object(ids_gathered, table['ids'])

            table['chexbert_y_hat'] = [j for i in chexbert_y_hat_gathered for j in i]
            table['chexbert_y'] = [j for i in chexbert_y_gathered for j in i]
            table['y_hat'] = [j for i in y_hat_gathered for j in i]
            table['y'] = [j for i in y_gathered for j in i]
            table['ids'] = [j for i in ids_gathered for j in i]

        columns = CONDITIONS + ['ids']
        df_y_hat = pd.DataFrame.from_records(table['chexbert_y_hat'], columns=columns)
        df_y = pd.DataFrame.from_records(table['chexbert_y'], columns=columns)

        df_y_hat = df_y_hat.drop_duplicates(subset=['ids'])
        df_y = df_y.drop_duplicates(subset=['ids'])

        df_y_hat = df_y_hat.drop(['ids'], axis=1)
        df_y = df_y.drop(['ids'], axis=1)

        df_y_hat = (df_y_hat == 1)
        df_y = (df_y == 1)

        tp = (df_y_hat * df_y).astype(float)

        fp = (df_y_hat * ~df_y).astype(float)
        fn = (~df_y_hat * df_y).astype(float)

        tp_cls = tp.sum()
        fp_cls = fp.sum()
        fn_cls = fn.sum()

        tp_eg = tp.sum(1)
        fp_eg = fp.sum(1)
        fn_eg = fn.sum(1)

        precision_class = (tp_cls / (tp_cls + fp_cls)).fillna(0)
        recall_class = (tp_cls / (tp_cls + fn_cls)).fillna(0)
        f1_class = (tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls))).fillna(0)

        scores = {
            'ce_precision_macro': precision_class.mean(),
            'ce_recall_macro': recall_class.mean(),
            'ce_f1_macro': f1_class.mean(),
            'ce_precision_micro': tp_cls.sum() / (tp_cls.sum() + fp_cls.sum()),
            'ce_recall_micro': tp_cls.sum() / (tp_cls.sum() + fn_cls.sum()),
            'ce_f1_micro': tp_cls.sum() / (tp_cls.sum() + 0.5 * (fp_cls.sum() + fn_cls.sum())),
            'ce_precision_example': (tp_eg / (tp_eg + fp_eg)).fillna(0).mean(),
            'ce_recall_example': (tp_eg / (tp_eg + fn_eg)).fillna(0).mean(),
            'ce_f1_example': (tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).fillna(0).mean(),
            'ce_num_examples': float(len(df_y_hat)),
        }

        if self.save_class_scores:
            save_path = enumerated_save_path(self.exp_dir, 'ce_class_metrics', '.csv')
            class_scores_dict = {
                **{'ce_precision_' + k: v for k, v in precision_class.to_dict().items()},
                **{'ce_recall_' + k: v for k, v in recall_class.to_dict().items()},
                **{'ce_f1_' + k: v for k, v in f1_class.to_dict().items()},
            }
            pd.DataFrame(class_scores_dict, index=['i', ]).to_csv(save_path, index=False)

        if self.save_outputs:

            def save():
                df = pd.DataFrame(table)
                df.chexbert_y_hat = [i[:-1] for i in df.chexbert_y_hat]
                df.chexbert_y = [i[:-1] for i in df.chexbert_y]
                df.to_csv(
                    os.path.join(self.exp_dir, 'chexbert_outputs_' + time.strftime("%d-%m-%Y_%H-%M-%S") + '.csv'),
                    index=False,
                    sep=';',
                )

            if not torch.distributed.is_initialized():
                save()
            elif torch.distributed.get_rank() == 0:
                save()

        return scores


class RadGraphMetrics(NaturalLanguage):
    is_differentiable = False
    full_state_update = False

    def __init__(
            self,
            radgraph_path,
            mbatch_size=16,
            save_class_scores=False,
            save_outputs=False,
            exp_dir=None,
    ):
        super().__init__(dist_sync_on_step=False)
        self.radgraph_path = radgraph_path
        self.mbatch_size = mbatch_size
        # self.save_class_scores = save_class_scores
        # self.save_outputs = save_outputs
        # self.exp_dir = exp_dir

    def mini_batch(self, iterable, mbatch_size=1):
        length = len(iterable)
        for i in range(0, length, mbatch_size):
            yield iterable[i:min(i + mbatch_size, length)]

    def compute(self):
        f1radgraph_partial = F1RadGraph(reward_level='partial',
                                        model_path=self.radgraph_path).to(self.device)

        table = {'hyps': [], 'refs': [], 'ids': []}
        for i in self.mini_batch(self.pairs, self.mbatch_size):
            y_hat, y, ids = zip(*i)
            table['hyps'].extend(y_hat)
            table['refs'].extend(y)
            table['ids'].extend(ids)

        if torch.distributed.is_initialized():  # If DDP

            y_hat_gathered = [None] * torch.distributed.get_world_size()
            y_gathered = [None] * torch.distributed.get_world_size()
            ids_gathered = [None] * torch.distributed.get_world_size()

            torch.distributed.all_gather_object(y_hat_gathered, table['hyps'])
            torch.distributed.all_gather_object(y_gathered, table['refs'])
            torch.distributed.all_gather_object(ids_gathered, table['ids'])

            # table['y_hat'] = [j for i in y_hat_gathered for j in i]
            # table['y'] = [j for i in y_gathered for j in i]
            # table['ids'] = [j for i in ids_gathered for j in i]

        # columns = CONDITIONS + ['ids']
        # df_y_hat = pd.DataFrame.from_records(table['y_hat'], columns=columns)
        # df_y = pd.DataFrame.from_records(table['y'], columns=columns)
        #
        # df_y_hat = df_y_hat.drop_duplicates(subset=['ids'])
        # df_y = df_y.drop_duplicates(subset=['ids'])

        partial_mean_reward, reward, hypothesis_ann, reference_ann = f1radgraph_partial(hyps=table['hyps'],
                                                                                        refs=table['refs'])

        scores = {
            "F1-Radgraph-partial": partial_mean_reward,
        }

        return scores


class F1CheXbertMetrics(NaturalLanguage):
    is_differentiable = False
    full_state_update = False

    def __init__(
            self,
            chexbert_path,
            model_path,
            mbatch_size=16,
            save_class_scores=False,
            save_outputs=False,
            exp_dir=None,
    ):
        super().__init__(dist_sync_on_step=False)
        self.chexbert_path = chexbert_path
        self.model_path = model_path
        self.mbatch_size = mbatch_size

    def mini_batch(self, iterable, mbatch_size=1):
        length = len(iterable)
        for i in range(0, length, mbatch_size):
            yield iterable[i:min(i + mbatch_size, length)]

    def compute(self):
        f1chexbert = F1CheXbert(chexbert_checkpoint=self.chexbert_path, model_checkpoint=self.model_path,
                                tokenizer_checkpoint=self.model_path).to(self.device)

        table = {'hyps': [], 'refs': [], 'ids': []}
        for i in self.mini_batch(self.pairs, self.mbatch_size):
            y_hat, y, ids = zip(*i)
            table['hyps'].extend(y_hat)
            table['refs'].extend(y)
            table['ids'].extend(ids)

        if torch.distributed.is_initialized():  # If DDP

            y_hat_gathered = [None] * torch.distributed.get_world_size()
            y_gathered = [None] * torch.distributed.get_world_size()
            ids_gathered = [None] * torch.distributed.get_world_size()

            torch.distributed.all_gather_object(y_hat_gathered, table['hyps'])
            torch.distributed.all_gather_object(y_gathered, table['refs'])
            torch.distributed.all_gather_object(ids_gathered, table['ids'])

        accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = f1chexbert(hyps=table['hyps'],
                                                                             refs=table['refs'])
        # default is chexbert_5_micro_f1
        chexbert_5_micro_f1 = chexbert_5["micro avg"]
        chexbert_all_micro_f1 = chexbert_all["micro avg"]
        chexbert_5_macro_f1 = chexbert_5["macro avg"]
        chexbert_all_macro_f1 = chexbert_all["macro avg"]

        scores = {
            "chexbert_5_micro_f1": chexbert_5_micro_f1["f1-score"],
            "chexbert_5_macro_f1": chexbert_5_macro_f1["f1-score"],
            "chexbert_all_micro_p": chexbert_all_micro_f1['precision'],
            "chexbert_all_micro_r": chexbert_all_micro_f1['recall'],
            "chexbert_all_micro_f1": chexbert_all_micro_f1["f1-score"],
            "chexbert_all_macro_p": chexbert_all_macro_f1['precision'],
            "chexbert_all_macro_r": chexbert_all_macro_f1['recall'],
            "chexbert_all_macro_f1": chexbert_all_macro_f1["f1-score"],
            "chexbert_accuracy": accuracy
        }

        return scores

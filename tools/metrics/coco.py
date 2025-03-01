from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from torchmetrics import Metric
import csv
import os
import pandas as pd
import re
import torch

# For _apply()
from collections.abc import Sequence
from typing import Callable
from torch import Tensor
from torch.nn import Module


class COCOCaptionMetrics(Metric):
    is_differentiable = False
    full_state_update = False

    def __init__(
            self,
            metrics=None,
            save=False,
            save_individual_scores=False,
            save_bootstrapped_scores=False,
            exp_dir=None,
            dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if metrics is None:
            metrics = ["bleu", "cider", "meteor", "rouge", "spice"]
        self.add_state("predictions", default=[])
        self.add_state("labels", default=[])
        self.add_state("ids", default=[])

        self.metrics = [metric.lower() for metric in metrics]
        self.save = save
        self.save_individual_scores = save_individual_scores
        self.save_bootstrapped_scores = save_bootstrapped_scores
        self.exp_dir = exp_dir

        self.num_metrics = 0
        if "bleu" in self.metrics:
            self.bleu = Bleu(4)
            self.num_metrics += 4
        if "meteor" in self.metrics:
            self.meteor = Meteor()
            self.num_metrics += 1
        if "rouge" in self.metrics:
            self.rouge = Rouge()
            self.num_metrics += 1
        if "cider" in self.metrics:
            self.cider = Cider()
            self.num_metrics += 1
        if "spice" in self.metrics:
            self.spice = Spice()
            self.num_metrics += 1

    def update(self, predictions, labels, ids):
        """
        Argument/s:
            predictions - the predicted captions must be in the following format:

                [
                    "a person on the snow practicing for a competition",
                    "group of people are on the side of a snowy field",
                ]

            labels - the corresponding labels must be in the following format:

                [
                    [
                        "Persons skating in the ice skating rink on the skateboard.",
                        "A snowboard sliding very gently across the snow in an enclosure.",
                        "A person on a snowboard getting ready for competition.",
                        "Man on snowboard riding under metal roofed enclosed area.",
                        "A snowboarder practicing his moves at a snow facility.",
                    ],
                    [
                        "There are mountains in the background and a lake in the middle.",
                        "a red fire hydrant in a field covered in snow",
                        "A fire hydrant in front of a snow covered field, a lake and
                        mountain backdrop.",
                        "A hydran in a snow covered field overlooking a lake.",
                        "An expanse of snow in the middle of dry plants",
                    ]
                ]

                or, if there is only one label per example (can still be in the above format):

                [
                    "Persons skating in the ice skating rink on the skateboard.",
                    "There are mountains in the background and a lake in the middle.",
                ]
            ids (list) - list of identifiers.
        """
        self.predictions.extend(list(predictions))
        self.labels.extend(list(labels))
        self.ids.extend(list(ids))

    def compute(self):
        """
        Compute the metrics from the COCO captioning task with and without DDP.

        Argument/s:
            stage - "val" or "test" stage of training.

        Returns:
            Dictionary containing the scores for each of the metrics
        """

        if torch.distributed.is_initialized():  # If DDP
            predictions_gathered = [None] * torch.distributed.get_world_size()
            labels_gathered = [None] * torch.distributed.get_world_size()
            ids_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(predictions_gathered, self.predictions)
            torch.distributed.all_gather_object(labels_gathered, self.labels)
            torch.distributed.all_gather_object(ids_gathered, self.ids)
            self.predictions = [j for i in predictions_gathered for j in i]
            self.labels = [j for i in labels_gathered for j in i]
            self.ids = [j for i in ids_gathered for j in i]

        return self.score()

    def score(self):

        predictions, labels = {}, {}
        for i, j, k in zip(self.ids, self.predictions, self.labels):
            predictions[i] = [re.sub(' +', ' ', j.replace(".", " ."))]
            labels[i] = [re.sub(' +', ' ', k.replace(".", " ."))]
        accumulated_scores = {}
        example_scores = {}
        if "bleu" in self.metrics:
            score, scores = self.bleu.compute_score(labels, predictions)
            accumulated_scores["chen_bleu_1"] = score[0]
            accumulated_scores["chen_bleu_2"] = score[1]
            accumulated_scores["chen_bleu_3"] = score[2]
            accumulated_scores["chen_bleu_4"] = score[3]
            example_scores["chen_bleu_1"] = scores[0]
            example_scores["chen_bleu_2"] = scores[1]
            example_scores["chen_bleu_3"] = scores[2]
            example_scores["chen_bleu_4"] = scores[3]
        if "meteor" in self.metrics:
            score, scores = self.meteor.compute_score(labels, predictions)
            accumulated_scores["chen_meteor"] = score
            example_scores["chen_meteor"] = scores
        if "rouge" in self.metrics:
            score, scores = self.rouge.compute_score(labels, predictions)
            accumulated_scores["chen_rouge"] = score
            example_scores["chen_rouge"] = scores
        if "cider" in self.metrics:
            score, scores = self.cider.compute_score(labels, predictions)
            accumulated_scores["chen_cider"] = score
            example_scores["chen_cider"] = scores
        if "spice" in self.metrics:
            score, scores = self.spice.compute_score(labels, predictions)
            accumulated_scores["chen_spice"] = score
            example_scores["chen_spice"] = scores
        accumulated_scores["chen_num_examples"] = len(predictions)

        if self.save:
            def save_reports():
                with open(os.path.join(self.exp_dir, "predictions.csv"), "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["prediction", "label", "id"])
                    for row in zip(predictions.values(), labels.values(), self.ids):
                        writer.writerow([row[0][0], row[1][0], row[2]])

            if not torch.distributed.is_initialized():
                save_reports()
            elif torch.distributed.get_rank() == 0:
                save_reports()

        if self.save_individual_scores:
            def save_example_scores():
                df = pd.DataFrame(example_scores)
                df.to_csv(os.path.join(self.exp_dir, "individual_scores.csv"))

            if not torch.distributed.is_initialized():
                save_example_scores()
            elif torch.distributed.get_rank() == 0:
                save_example_scores()

        if self.save_bootstrapped_scores:
            df = pd.DataFrame(accumulated_scores, index=[0, ])
            save_path = os.path.join(self.exp_dir, "bootstrapped_scores.csv")
            header = False if os.path.isfile(save_path) else True
            df.to_csv(save_path, mode="a", header=header, index=False)

        return accumulated_scores

    # # Need this for non-ddp at the moment as torch can't handle lists of strings or dicts.
    # def _apply(self, fn: Callable) -> Module:
    #     """Overwrite _apply function such that we can also move metric states
    #     to the correct device when `.to`, `.cuda`, etc methods are called
    #     """
    #     this = super(Metric, self)._apply(fn)

    #     self.labels.clear()
    #     self.predictions.clear()
    #     self.ids.clear()

    #     # Also apply fn to metric states and defaults
    #     for key, value in this._defaults.items():
    #         if isinstance(value, Tensor):
    #             this._defaults[key] = fn(value)
    #         elif isinstance(value, Sequence):
    #             this._defaults[key] = [fn(v) for v in value]

    #         current_val = getattr(this, key)
    #         if isinstance(current_val, Tensor):
    #             setattr(this, key, fn(current_val))
    #         elif isinstance(current_val, Sequence):
    #             setattr(this, key, [fn(cur_v) for cur_v in current_val])
    #         else:
    #             raise TypeError(
    #                 "Expected metric state to be either a Tensor"
    #                 f"or a list of Tensor, but encountered {current_val}"
    #             )
    #     return this

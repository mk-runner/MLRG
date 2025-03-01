from torchmetrics import Metric
import pandas as pd
import torch


class NaturalLanguage(Metric):

    is_differentiable = False
    higher_is_better = True

    def __init__(self, dist_sync_on_step=False):

        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('pairs', default=[])  # prediction and label pairs.

    def update(self, y_hat, y, ids):
        self.pairs.extend(list(zip(y_hat, y, ids)))

    def compute(self):

        if torch.distributed.is_initialized():  # If DDP
            predictions_gathered = [None] * torch.distributed.get_world_size()
            labels_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(predictions_gathered, self.predictions)
            torch.distributed.all_gather_object(labels_gathered, self.labels)
            self.predictions = [j for i in predictions_gathered for j in i]
            self.labels = [j for i in labels_gathered for j in i]

        df = pd.DataFrame.from_records(self.pairs, columns=['y_hat', 'y', 'ids']).drop_duplicates()

        self.scores(df)

    def scores(self, df: pd.DataFrame):
        raise NotImplementedError

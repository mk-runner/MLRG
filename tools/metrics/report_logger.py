from pathlib import Path
from torchmetrics import Metric
import os
import pandas as pd
import time
import torch


class ReportLogger(Metric):

    is_differentiable = False
    full_state_update = False

    """
    Logs the reports to a .csv.
    """

    def __init__(
            self,
            exp_dir: str,
            split: str,
            dist_sync_on_step: bool = False,
    ):
        """
        exp_dir - experiment directory to save the captions and individual scores.
        split - train, val, or test split.
        dist_sync_on_step - sync the workers at each step.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.exp_dir = exp_dir
        self.split = split

        # No dist_reduce_fx, manually sync over devices
        self.add_state('reports', default=[])

        self.save_dir = os.path.join(self.exp_dir, 'generated_reports')

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def update(self, reports, dicom_ids=None, labels=None):
        """
        Argument/s:
            report - the report must be in the following format:

                [
                    '...',
                    '...',
                ]
            dicom_ids - list of dicom identifiers.
        """

        assert isinstance(reports, list), '"reports" must be a list of strings.'
        assert all(isinstance(i, str) for i in reports), 'Each element of "reports" must be a string.'
        if labels is None:
            for (i, j) in zip(reports, dicom_ids):
                self.reports.append({'dicom_id': j, 'report': i})
        else:
            for (i, j, l) in zip(reports, dicom_ids, labels):
                self.reports.append({'dicom_id': j, "labels": l, 'report': i})

    def compute(self, epoch):
        if torch.distributed.is_initialized():  # If DDP
            reports_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(reports_gathered, self.reports)
            self.reports = [j for i in reports_gathered for j in i]

        return self.log(epoch)

    def log(self, epoch):

        def save():
            df = pd.DataFrame(self.reports).drop_duplicates(subset='dicom_id')

            df.to_csv(
                os.path.join(self.save_dir, f'{self.split}_epoch-{epoch}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save()
        elif torch.distributed.get_rank() == 0:
            save()

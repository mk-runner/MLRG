import os
import torch
import logging
import torch.nn as nn
import numpy as np

from radgraph.allennlp.commands.predict import _PredictManager
from radgraph.allennlp.common.plugins import import_plugins
from radgraph.allennlp.common.util import import_module_and_submodules
from radgraph.allennlp.predictors.predictor import Predictor
from radgraph.allennlp.models.archival import load_archive
from radgraph.allennlp.common.checks import check_for_gpu

logging.getLogger("radgraph").setLevel(logging.CRITICAL)
logging.getLogger("allennlp").setLevel(logging.CRITICAL)

from radgraph.utils import download_model
from radgraph.utils import (
    preprocess_reports,
    postprocess_reports,
)

from radgraph.rewards import compute_reward
# from appdirs import user_cache_dir

# CACHE_DIR = user_cache_dir("radgraph")
# CACHE_DIR = r'D:\Code\checkpoints/'



class RadGraph(nn.Module):
    def __init__(
            self,
            batch_size=1,
            cuda=None,
            model_path=None,
            **kwargs
    ):

        super().__init__()
        if cuda is None:
            cuda = 0 if torch.cuda.is_available() else -1
        self.cuda = cuda
        self.batch_size = batch_size
        self.model_path = model_path

        try:
            if not os.path.exists(self.model_path):
                download_model(
                    repo_id="StanfordAIMI/RRG_scorers",
                    cache_dir=CACHE_DIR,
                    filename="radgraph.tar.gz",
                )
        except Exception as e:
            print("Model download error", e)

        # Model
        import_plugins()
        import_module_and_submodules("radgraph.dygie")

        check_for_gpu(self.cuda)
        archive = load_archive(
            self.model_path,
            weights_file=None,
            cuda_device=self.cuda,
            overrides="",
        )
        self.predictor = Predictor.from_archive(
            archive, predictor_name="dygie", dataset_reader_to_load="validation"
        )

    def forward(self, hyps):

        assert isinstance(hyps, str) or isinstance(hyps, list)
        if isinstance(hyps, str):
            hyps = [hyps]

        hyps = ["None" if not s else s for s in hyps]

        # Preprocessing
        model_input = preprocess_reports(hyps)
        # AllenNLP
        manager = _PredictManager(
            predictor=self.predictor,
            input_file=str(
                model_input
            ),  # trick the manager, make the list as string so it thinks its a filename
            output_file=None,
            batch_size=self.batch_size,
            print_to_console=False,
            has_dataset_reader=True,
        )
        results = manager.run()

        # Postprocessing
        inference_dict = postprocess_reports(results)
        return inference_dict


class F1RadGraph(nn.Module):
    def __init__(
            self,
            reward_level,
            **kwargs
    ):

        super().__init__()
        assert reward_level in ["simple", "partial", "complete", "all"]
        self.reward_level = reward_level
        self.radgraph = RadGraph(**kwargs)

    def forward(self, refs, hyps):
        # Checks
        assert isinstance(hyps, str) or isinstance(hyps, list)
        assert isinstance(refs, str) or isinstance(refs, list)

        if isinstance(hyps, str):
            hyps = [hyps]
        if isinstance(hyps, str):
            refs = [refs]

        assert len(refs) == len(hyps)

        # getting empty report list
        number_of_reports = len(hyps)
        empty_report_index_list = [i for i in range(number_of_reports) if (len(hyps[i]) == 0) or (len(refs[i]) == 0)]
        number_of_non_empty_reports = number_of_reports - len(empty_report_index_list)

        # stacking all reports (hyps and refs)
        report_list = [
                          hypothesis_report
                          for i, hypothesis_report in enumerate(hyps)
                          if i not in empty_report_index_list
                      ] + [
                          reference_report
                          for i, reference_report in enumerate(refs)
                          if i not in empty_report_index_list
                      ]

        assert len(report_list) == 2 * number_of_non_empty_reports

        # getting annotations
        inference_dict = self.radgraph(report_list)

        # Compute reward
        reward_list = []
        hypothesis_annotation_lists = []
        reference_annotation_lists = []
        non_empty_report_index = 0
        for report_index in range(number_of_reports):
            if report_index in empty_report_index_list:
                if self.reward_level == "all":
                    reward_list.append((0., 0., 0.))
                else:
                    reward_list.append(0.)
                continue

            hypothesis_annotation_list = inference_dict[str(non_empty_report_index)]
            reference_annotation_list = inference_dict[
                str(non_empty_report_index + number_of_non_empty_reports)
            ]

            reward_list.append(
                compute_reward(
                    hypothesis_annotation_list,
                    reference_annotation_list,
                    self.reward_level,
                )
            )
            reference_annotation_lists.append(reference_annotation_list)
            hypothesis_annotation_lists.append(hypothesis_annotation_list)
            non_empty_report_index += 1

        assert non_empty_report_index == number_of_non_empty_reports

        if self.reward_level == "all":
            reward_list = ([r[0] for r in reward_list], [r[1] for r in reward_list], [r[2] for r in reward_list])
            mean_reward = (np.mean(reward_list[0]), np.mean(reward_list[1]), np.mean(reward_list[2]))
        else:
            mean_reward = np.mean(reward_list)

        return (
            mean_reward,
            reward_list,
            hypothesis_annotation_lists,
            reference_annotation_lists,
        )


# if __name__ == "__main__":
#     radgraph = RadGraph(model_path=r'D:\Code\checkpoints\radgraph')
#     annotations = radgraph(["no evidence of acute cardiopulmonary process moderate hiatal hernia"])
#     print(annotations)
#
#     # {'0': {'text': 'no evidence of acute cardiopulmonary process moderate hiatal hernia',
#     #        'entities': {'1': {'tokens': 'acute', 'label': 'OBS-DA', 'start_ix': 3, 'end_ix': 3, 'relations': []},
#     #                     '2': {'tokens': 'cardiopulmonary', 'label': 'ANAT-DP', 'start_ix': 4, 'end_ix': 4,
#     #                           'relations': []},
#     #                     '3': {'tokens': 'process', 'label': 'OBS-DA', 'start_ix': 5, 'end_ix': 5,
#     #                           'relations': [['located_at', '2']]},
#     #                     '4': {'tokens': 'moderate', 'label': 'OBS-DP', 'start_ix': 6, 'end_ix': 6, 'relations': []},
#     #                     '5': {'tokens': 'hiatal', 'label': 'ANAT-DP', 'start_ix': 7, 'end_ix': 7, 'relations': []},
#     #                     '6': {'tokens': 'hernia', 'label': 'OBS-DP', 'start_ix': 8, 'end_ix': 8, 'relations': []}},
#     #        'data_source': None, 'data_split': 'inference'}}
#
#     refs = ["no acute cardiopulmonary abnormality",
#             "et tube terminates 2 cm above the carina retraction by several centimeters is recommended for more optimal placement bibasilar consolidations better assessed on concurrent chest ct",
#             "there is no significant change since the previous exam the feeding tube and nasogastric tube have been removed",
#             "unchanged mild pulmonary edema no radiographic evidence pneumonia",
#             "no evidence of acute pulmonary process moderately large size hiatal hernia",
#             "no acute intrathoracic process"]
#
#     hyps = ["no acute cardiopulmonary abnormality",
#             "endotracheal tube terminates 2 5 cm above the carina bibasilar opacities likely represent atelectasis or aspiration",
#             "there is no significant change since the previous exam",
#             "unchanged mild pulmonary edema and moderate cardiomegaly",
#             "no evidence of acute cardiopulmonary process moderate hiatal hernia",
#             "no acute cardiopulmonary process"]
#
#     f1radgraph = F1RadGraph(reward_level="all", model_path=r'D:\Code\checkpoints\radgraph')
#     mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=hyps,
#                                                                                                    refs=refs)
#     print(mean_reward)
#     print(reward_list)
    # (0.6238095238095238, 0.5111111111111111, 0.5011204481792717)
    # ([1.0, 0.4, 0.5714285714285715, 0.8, 0.5714285714285715, 0.4],
    #  [1.0, 0.26666666666666666, 0.5714285714285715, 0.4, 0.42857142857142855, 0.4],
    #  [1.0, 0.23529411764705885, 0.5714285714285715, 0.4, 0.4, 0.4])

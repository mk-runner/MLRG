# Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation
[arXiv](https://arxiv.org/abs/2502.20056)

# News
-  **2025-03-01** upload the code, the [generated reports](scripts/results/mimic_cxr/test/ft_100_top1/) for the MIMIC-CXR test set.

## Requirements

- `torch==2.1.2+cu118`
- `transformers==4.23.1`
- `torchvision==0.16.2+cu118`
- `radgraph==0.09`
- please refer to `requirements.txt` for more details.

## Checkpoints

- Checkpoints (pretrain and finetune) and logs for the MIMIC-CXR dataset are available at [Baidu Netdisk]([https://pan.baidu.com/s/15SW1k3xZ57S06FUeqpclAA](https://pan.baidu.com/s/1Rnwc1ZKhcieBjHoXpHTnlw?pwd=MK13 )) and [huggingface 🤗](https://huggingface.co/MK-runner/MLRG).

## Datasets

- Medical images from the MIMIC-CXR, MIMIC-ABN, and Two-view CXR datasets are available for download from [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/) and [NIH](https://openi.nlm.nih.gov/faq#collection), with NIH data used exclusively in the Two-view CXR dataset. The file structure for storing these images is as follows:  

```
files/
├── p10
├── p11
├── p12
├── p13
├── p14
├── p15
├── p16
├── p17
├── p18
├── p19
└── NLMCXR_png
```
- The radiology reports for MIMIC-CXR, MIMIC-ABN, and Two-view CXR are available on [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/), [NIH](https://openi.nlm.nih.gov/faq#collection), and [huggingface 🤗](https://huggingface.co/datasets/MK-runner/Multi-view-CXR), respectively. To streamline usage, we have structured multi-view longitudinal data by the `study_id`. The processed data can be accessed on [huggingface 🤗](https://huggingface.co/MK-runner/MLRG) (PhysioNet authorization required).

## Reproducibility on MIMIC-CXR

### Structural entities extraction (SEE) approach

1. Config RadGraph environment based on `knowledge_encoder/factual_serialization.py`


   ===================environmental setting=================
   
    Basic Setup (One-time activity)

   a. Clone the DYGIE++ repository from [here](https://github.com/dwadden/dygiepp). This repository is managed by Wadden et al., authors of the paper [Entity, Relation, and Event Extraction with Contextualized Span Representations](https://www.aclweb.org/anthology/D19-1585.pdf).
    ```bash
   git clone https://github.com/dwadden/dygiepp.git
    ```
    
   b. Navigate to the root of repo in your system and use the following commands to set the conda environment:
    ```bash
   conda create --name dygiepp python=3.7
   conda activate dygiepp
   cd dygiepp
   pip install -r requirements.txt
   conda develop .   # Adds DyGIE to your PYTHONPATH
   ```

   c. Activate the conda environment:
    
    ```bash
   conda activate dygiepp
    ```
    Notably, for our RadGraph environment, you can refer to `knowledge_encoder/radgraph_requirements.yml`.
   
2. Config `radgraph_path` and `ann_path` in `knowledge_encoder/see.py`. `annotation.json`, can be obtained from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing). Note that you can apply with your license of [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

3. Run the `knowledge_encoder/see.py` to extract factual entity sequence for each report.
   
4. Finally, the `annotation.json` becomes `mimic_cxr_annotation_sen.json` that is identical to `new_ann_file_name` variable in `see.py`


### Conducting the first stage (i.e., training cross-modal alignment module)

1. Run `bash pretrain_mimic_cxr.sh` to pretrain a model on the MIMIC-CXR data (Note that the `mimic_cxr_ann_path` is `mimic_cxr_annotation_sen.json`).

### Similar historical cases retrieval for each sample

1. Config `--load` argument in `pretrain_inference_mimic_cxr.sh`. Note that the argument is the pre-trained model from the first stage.

2. Run `bash pretrain_inference_mimic_cxr.sh` to retrieve similar historical cases for each sample, forming `mimic_cxr_annotation_sen_best_reports_keywords_20.json` (i.e., the `mimic_cxr_annotation_sen.json` becomes this `*.json` file).

### Conducting the second stage (i.e., training report generation module)


1. Extract and preprocess the `indication section` in the radiology report.

   a. Config `ann_path` and `report_dir` in `knowledge_encoder/preprocessing_indication_section.py`, and its value is `mimic_cxr_annotation_sen_best_reports_keywords_20.json`. 
      Note that `report_dir` can be downloaded from [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/).
   
   b. Run `knowledge_encoder/preprocessing_indication_section.py`, forming `mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json`


2. Config `--load` argument in `finetune_mimic_cxr.sh`. Note that the argument is the pre-trained model from the first stage. Furthermore, `mimic_cxr_ann_path` is `mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json`

3. Download these checkpoints. Notably, the `chexbert.pth` and `radgraph` are used to calculate CE metrics, and `bert-base-uncased` and `scibert_scivocab_uncased ` are pre-trained models for cross-modal fusion network and text encoder. Then put these checkpoints in the same local dir (e.g., "/home/data/checkpoints"), and configure the `--ckpt_zoo_dir /home/data/checkpoints` argument in `finetune_mimic_cxr.sh`

<div style="margin: 0 auto; width: fit-content;">
      
| **Chekpoint**                    | **Variable\_name** | **Download**                                                                          |
| :------------------------------- | :----------------- | :------------------------------------------------------------------------------------ |
| chexbert.pth                     | chexbert\_path     | [here](https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9)       |
| bert-base-uncased                | bert\_path         | [huggingface](https://huggingface.co/google-bert/bert-base-uncased)                   |
| radgraph                         | radgraph\_path     | [PhysioNet](https://physionet.org/content/radgraph/1\.0.0/)                           |
| scibert\_scivocab\_uncased       | scibert\_path      | [huggingface](https://huggingface.co/allenai/scibertsscivocabuuncased)                |

</div>

4. Run `bash finetune_mimic_cxr.sh` to generate reports based on similar historical cases.


### Test 

1. You must download the medical images, their corresponding reports (i.e., `mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json`),  and checkpoints (i.e., `SEI-1-finetune-model-best.pth`) in Section Datasets and Section Checkpoints, respectively.

2. Config `--load` and `--mimic_cxr_ann_path`arguments in `test_mimic_cxr.sh`

3. Run `bash test_mimic_cxr.sh` to generate reports based on similar historical cases.

4. Results on MIMIC-CXR are presented as follows:

<div align=center><img src="results/sei_on_mimic_cxr.jpg"></div>


5. Next, the code for this project will be streamlined.


# Experiments
## Main Results
<div align=center><img src="results/main_results.jpg"></div>

## Ablation Study
<div align=center><img src="results/ablation_study.jpg"></div>
<div align=center><img src="results/fig2.jpg"></div>


## Citations

If you use or extend our work, please cite our paper at MICCAI 2024.

```
@InProceedings{liu-sei-miccai-2024,
      author={Liu, Kang and Ma, Zhuoqi and Kang, Xiaolu and Zhong, Zhusi and Jiao, Zhicheng and Baird, Grayson and Bai, Harrison and Miao, Qiguang},
      title={Structural Entities Extraction and Patient Indications Incorporation for Chest X-Ray Report Generation},
      booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
      year={2024},
      publisher={Springer Nature Switzerland},
      address={Cham},
      pages={433--443},
      isbn={978-3-031-72384-1},
      doi={10.1007/978-3-031-72384-1_41}
}

```


## Acknowledgement

- [R2Gen](https://github.com/zhjohnchan/R2Gen) Some codes are adapted based on R2Gen.
- [R2GenCMN](https://github.com/zhjohnchan/R2GenCMN) Some codes are adapted based on R2GenCMN.
- [MGCA](https://github.com/HKU-MedAI/MGCA) Some codes are adapted based on MGCA.

## References

[1] Chen, Z., Song, Y., Chang, T.H., Wan, X., 2020. Generating radiology reports via memory-driven transformer, in: EMNLP, pp. 1439–1449. 

[2] Chen, Z., Shen, Y., Song, Y., Wan, X., 2021. Cross-modal memory networks for radiology report generation, in: ACL, pp. 5904–5914. 

[3] Wang, F., Zhou, Y., Wang, S., Vardhanabhuti, V., Yu, L., 2022. Multigranularity cross-modal alignment for generalized medical visual representation learning, in: NeurIPS, pp. 33536–33549.


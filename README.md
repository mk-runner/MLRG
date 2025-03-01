# Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation
[![arXiv](https://img.shields.io/badge/arXiv-2502.20056-b31b1b.svg)](https://arxiv.org/abs/2502.20056)

# News
-  **2025-03-01** Upload the code, checkpoints, and the [generated radiology reports](generated-radiology-reports) for the MIMIC-CXR, MIMIC-ABN, and Two-view CXR datasets. Notably, in the **generated-radiology-reports**, the **labels** column corresponds to **reference reports**, while the **report** column represents **generated reports**.

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
- The radiology reports for MIMIC-CXR, MIMIC-ABN, and Two-view CXR are available on [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/), [NIH](https://openi.nlm.nih.gov/faq#collection), and [huggingface 🤗](https://huggingface.co/datasets/MK-runner/Multi-view-CXR), respectively. To streamline usage, we have structured multi-view longitudinal data using the `study_id`. The processed data can be accessed on [huggingface 🤗](https://huggingface.co/MK-runner/MLRG) (PhysioNet authorization required).

## Evaluation using generated radiology reports


3. Download checkpoints below. Notably, the `chexbert.pth`, `radgraph`, and `bert-base-uncased` are used to calculate CE metrics, and `bert-base-uncased` and `scibert_scivocab_uncased ` are pre-trained models for cross-modal fusion network and text encoder. Then put these checkpoints in the same local dir (e.g., "/home/data/checkpoints"), and configure the `--ckpt_zoo_dir /home/data/checkpoints` argument in `script/**/**.sh`

<div style="margin: 0 auto; width: fit-content;">
      
| **Chekpoint**                    | **Variable\_name** | **Download**                                                                          |
| :------------------------------- | :----------------- | :------------------------------------------------------------------------------------ |
| chexbert.pth                     | chexbert\_path     | [here](https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9)       |
| bert-base-uncased                | bert\_path         | [huggingface](https://huggingface.co/google-bert/bert-base-uncased)                   |
| radgraph                         | radgraph\_path     | [PhysioNet](https://physionet.org/content/radgraph/1\.0.0/)                           |
| scibert\_scivocab\_uncased       | scibert\_path      | [huggingface](https://huggingface.co/allenai/scibertsscivocabuuncased)                |

</div>


## Citations

If you use or extend our work, please cite our paper at CVPR 2025.

```
@misc{liu2025enhancedcontrastivelearningmultiview,
      title={Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation}, 
      author={Kang Liu and Zhuoqi Ma and Xiaolu Kang and Yunan Li and Kun Xie and Zhicheng Jiao and Qiguang Miao},
      year={2025},
      eprint={2502.20056},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.20056}, 
}
```


## Acknowledgement

- [cvt2distilgpt2](https://github.com/aehrc/cvt2distilgpt2) Some codes are adapted based on R2Gen.

## References

[1] Nicolson, A., Dowling, J., & Koopman, B. (2023). Improving chest X-ray report generation by leveraging warm starting. Artificial Intelligence in Medicine, 144, 102633. 

[2] Chen, Z., Shen, Y., Song, Y., Wan, X., 2021. Cross-modal memory networks for radiology report generation, in: ACL, pp. 5904–5914. 

[3] Wang, F., Zhou, Y., Wang, S., Vardhanabhuti, V., Yu, L., 2022. Multigranularity cross-modal alignment for generalized medical visual representation learning, in: NeurIPS, pp. 33536–33549.


<div align="center">

# Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation

[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-blue.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Enhanced_Contrastive_Learning_with_Multi-view_Longitudinal_Data_for_Chest_X-ray_CVPR_2025_paper.html)&nbsp;&nbsp;&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2502.20056-b31b1b.svg)](https://arxiv.org/abs/2502.20056)&nbsp;&nbsp;&nbsp;
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/MK-runner/MLRG)&nbsp;&nbsp;&nbsp;
[![BibTeX](https://img.shields.io/badge/%F0%9F%93%96-BibTeX-yellow)](#-Citation)



<div align="center">
  <img src="generated-radiology-reports/fig2.png" alt="Framework" width="75%">
</div>

</div>

---

## 📢 News

- **2025-05-09** &nbsp; Upload [**poster**](https://github.com/mk-runner/MLRG/tree/main/generated-radiology-reports/mlrg-poster.pdf)  
- **2025-03-16** &nbsp; Release checkpoints for [MIMIC-ABN](https://huggingface.co/MK-runner/MLRG/tree/main/mimic-abn) and [Two-view CXR](https://huggingface.co/MK-runner/MLRG/blob/main/two-view%20cxr/best_model.ckpt)  
- **2025-03-01** &nbsp; Upload official code and checkpoints for [MIMIC-CXR](https://huggingface.co/MK-runner/MLRG/tree/main/mimic-cxr)  
- **2025-03-01** &nbsp; Release [**generated-radiology-reports**](https://github.com/mk-runner/MLRG/tree/main/generated-radiology-reports) — **labels** = reference reports, **report** = generated reports  

---

## ⚙️ Requirements

```bash
# create virtual environment
conda create -n mlrg python=3.9.0

# install dependencies
pip install -r requirements.txt
````

* `torch==2.3.1+cu118`
* `transformers==4.43.3` (As stated in [Issue #4](https://github.com/mk-runner/MLRG/issues/4), **the `transformers` version should be maintained to prevent potential problems.** Credit goes to [@Andy](https://github.com/andypinxinliu) for this clarification)
* `torchvision==0.18.1+cu118`
* `radgraph==0.09`
> Please refer to `requirements.txt` for more details.
---

## 📦 Checkpoints for our MLRG

| 📊 **Dataset**   | 🪣 **Download**                                                                                                   | 📄 **Generated Reports**                                                                                                                                   |
|------------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MIMIC-CXR`      | [HuggingFace](https://huggingface.co/MK-runner/MLRG/tree/main/mimic-cxr)                                          | [GitHub](https://github.com/mk-runner/MLRG/blob/main/generated-radiology-reports/MIMIC-CXR/test_reports_epoch-1_20-10-2024_16-28-28.csv)                   |
| `MIMIC-ABN`      | [HuggingFace](https://huggingface.co/MK-runner/MLRG/tree/main/mimic-abn)                                          | [GitHub](https://github.com/mk-runner/MLRG/blob/main/generated-radiology-reports/MIMIC-ABN/test_reports_epoch-1_23-10-2024_10-25-20.csv)                   |
| `Two-view CXR`   | [HuggingFace](https://huggingface.co/MK-runner/MLRG/tree/main/two-view%20cxr)                                     | [GitHub](https://github.com/mk-runner/MLRG/blob/main/generated-radiology-reports/Two-view%20CXR/test_reports_epoch-0_25-10-2024_11-38-35.csv)              |
---

## 📂 Datasets

### Medical Images

* **MIMIC-CXR / MIMIC-ABN** — [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/), with data systematically organized under root directories labeled `p10` through `p19`, maintaining consistency with MIMIC-CXR's default configuration.
* **IU X-ray** — [NIH](https://openi.nlm.nih.gov/faq#collection), its root directory is the `NLMCXR_png`.
* **Two-View CXR** — aggregated studies with two views from MIMIC-CXR + IU X-ray ([arXiv](https://arxiv.org/abs/2411.10224)&nbsp;&nbsp;&nbsp;[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/datasets/MK-runner/Multi-view-CXR))

```
files/
├── p10
    └── p10000032
            └── s50414267
               ├── 02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
               └── 174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg
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
   ├── CXR1_1_IM-0001-3001.png
   ├── CXR1_1_IM-0001-4001.png
   └── CXR2_IM-0652-1001.png
```
### Raw Radiology Reports
- MIMIC-CXR and MIMIC-ABN: [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/).
- Two-view CXR: [HuggingFace 🤗](https://huggingface.co/datasets/MK-runner/Multi-view-CXR).
  
### Reorganization of Raw Radiology Reports
-  To simplify usage, we have organized multi-view longitudinal data using the `study_id`. The processed datasets—MIMIC-CXR, MIMIC-ABN, and Two-view CXR—are available on [HuggingFace 🤗](https://huggingface.co/MK-runner/MLRG/tree/main/radiology%20report) (PhysioNet authorization required). Note that the IU X-ray dataset (`NLMCXR_png`) does not include previous visit data due to the absence of `study_id`.
- MIMIC-CXR: [five_work_mimic_cxr_annotation_v1.1.json](https://huggingface.co/MK-runner/MLRG/blob/main/radiology%20report/five_work_mimic_cxr_annotation_v1.1.json)
- MIMIC-ABN: [mlrg_mimic_abn_annotation_v1.1.json](https://huggingface.co/MK-runner/MLRG/blob/main/radiology%20report/mlrg_mimic_abn_annotation_v1.1.json)
- Two-view CXR: [mlrg_multiview_cxr_annotation_v1.1.json](https://huggingface.co/MK-runner/MLRG/blob/main/radiology%20report/mlrg_multiview_cxr_annotation_v1.1.json)
- View Position for all datasets: [five_work_mimic_cxr_view_position_v1.1.json](https://huggingface.co/MK-runner/MLRG/blob/main/radiology%20report/five_work_mimic_cxr_view_position_v1.1.json)


---

## 📊 Evaluation using generated radiology reports

```python
def compute_performance_using_generated_reports():
    from tools.metrics.metrics import compute_all_scores, compute_chexbert_details_scores
    mimic_cxr_generated_path = 'generated-radiology-reports/MIMIC-CXR/test_reports_epoch-1_20-10-2024_16-28-28.csv'
    mimic_abn_generated_path = 'generated-radiology-reports/MIMIC-ABN/test_reports_epoch-1_23-10-2024_10-25-20.csv'
    twoview_cxr_generated_path = 'generated-radiology-reports/Two-view CXR/test_reports_epoch-0_25-10-2024_11-38-35.csv'
    args = {
        'chexbert_path': "/home/miao/data/dataset/checkpoints/chexbert.pth",
        'bert_path': "/home/miao/data/dataset/checkpoints/bert-base-uncased",
        'radgraph_path': "/home/miao/data/dataset/checkpoints/radgraph",
    }
    for generated_path in [mimic_cxr_generated_path, mimic_abn_generated_path, twoview_cxr_generated_path]:
        data = pd.read_csv(generated_path)
        gts, gens = data['labels'].tolist(), data['report'].tolist()
        scores = compute_all_scores(gts, gens, args)
        print(scores)
```

---

## 🚀 Training

**1. Download checkpoints for architecture and metrics.**
- For CE metrics calculation: `chexbert.pth`, `radgraph`, and `bert-base-uncased`.
- For model initialization: `microsoft/rad-dino` (image encoder), `microsoft/BiomedVLP-CXR-BERT-specialized` (text encoder), `distilbert/distilgpt2` (define text generator), and `cvt2distilgpt2` (initialize text generator).
- Checkpoint directory: Place all checkpoints in a local directory (e.g., "/home/data/checkpoints"), and configure the `--ckpt_zoo_dir /home/data/checkpoints` argument in the corresponding `script/**/**.sh` file.

| **Checkpoint**                             | **Variable name**     | **Download**                                                                                                                                                               |
| ------------------------------------------ | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chexbert.pth`                             | `chexbert_path`       | [StanfordMedicine](https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) • [HuggingFace](https://huggingface.co/MK-runner/RRG-metrics-pretrained-model) |
| `bert-base-uncased`                        | `bert_path`           | [HuggingFace](https://huggingface.co/google-bert/bert-base-uncased)                                                                                                        |
| `radgraph`                                 | `radgraph_path`       | [PhysioNet](https://physionet.org/content/radgraph/1.0.0/)                                                                                                                 |
| `microsoft/rad-dino`                       | `rad_dino_path`       | [HuggingFace](https://huggingface.co/microsoft/rad-dino)                                                                                                                   |
| `microsoft/BiomedVLP-CXR-BERT-specialized` | `cxr_bert_path`       | [HuggingFace](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized)                                                                                             |
| `distilbert/distilgpt2`                    | `distilgpt2_path`     | [HuggingFace](https://huggingface.co/distilbert/distilgpt2)                                                                                                                |
| `cvt2distilgpt2`                           | `cvt2distilgpt2_path` | [GitHub](https://github.com/aehrc/cvt2distilgpt2)    


**2. Conducting Stages 1 and 2**
```bash

# Stage 1: Multi-view Longitudinal Contrastive Learning
cd script/MIMIC-CXR
bash run_cxr_pt_v0906_fs.sh

# Stage 2: Chest X-ray Report Generation based on Patient-specific Prior Knowledge
cd script/MIMIC-CXR
bash run_cxr_ft_mlrg_v1011.sh
```

---

## 📜 Citation

If you use or extend our work, please cite our paper at CVPR 2025.

```bibtex
@InProceedings{Liu_2025_CVPR,
    author    = {Liu, Kang and Ma, Zhuoqi and Kang, Xiaolu and Li, Yunan and Xie, Kun and Jiao, Zhicheng and Miao, Qiguang},
    title     = {Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation},
    booktitle = {CVPR},
    month     = {June},
    year      = {2025},
    pages     = {10348-10359}
}
```

---

## 🙏 Acknowledgements

* [cvt2distilgpt2](https://github.com/aehrc/cvt2distilgpt2) — adapted from R2Gen
* [EVOKE](https://github.com/mk-runner/EVOKE) — dataset & implementation references

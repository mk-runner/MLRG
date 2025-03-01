import os
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PretrainBaseDataset(Dataset):  # finetune and inference phase
    def __init__(self, args, split, tokenizer=None):
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        if args['report_style'] == 'factual_serialization':
            for item in ann:
                if len(item['findings_factual_serialization']) == 0:
                    continue   # delete invalid findings (which has not clinical meaning.)
                findings_fs = list(map(lambda x: str(x).strip().lower(), item['findings_factual_serialization']))
                self.examples.append({
                    'id': item['id'],
                    'anchor_scan': item['anchor_scan'],  # list
                    'auxiliary_references': item['auxiliary_references'],
                    'report': "[CLS]" + "[SEP]".join(findings_fs) + '[SEP]',
                    'prior_study': item['prior_study']

                })
        else:   # report
            for item in ann:
                if len(item['findings_factual_serialization']) == 0:
                    continue
                findings_fs = item['findings'].strip().lower()
                self.examples.append({
                    'id': item['id'],
                    'anchor_scan': item['anchor_scan'],  # list
                    'auxiliary_references': item['auxiliary_references'],
                    'report': "[CLS]" + findings_fs + "[SEP]",
                    'prior_study': item['prior_study']
                })
        # self.examples = self.examples[:4]

    def __len__(self):
        return len(self.examples)


class MimiccxrPretrainDataset(PretrainBaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        sample = (image_id, example['anchor_scan'], example['auxiliary_references'],
                  example['report'], example['prior_study'])
        return sample


class FinetuneBaseDataset(Dataset):  # finetune and inference phase
    def __init__(self, args, split, tokenizer=None):
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        for item in ann:
            if len(item['findings_factual_serialization']) == 0:
                continue  # delete invalid findings (which has not clinical meaning.)
            image_num = len(item['anchor_scan']['image_path']) + len(item['auxiliary_references']['image_path'])
            if args['data_name'] == 'twoview_cxr' and image_num != 2:
                continue
            indication = item['indication_pure'].strip().lower() if len(item['indication_pure']) != 0 else '[NHI]'
            self.examples.append({
                'id': item['id'],
                'anchor_scan': item['anchor_scan'],  # list
                'auxiliary_references': item['auxiliary_references'],
                'report': "[BOS]" + item['findings'].strip().lower() + "[EOS]",
                "indication": "[CLS]" + indication + '[SEP]',
                'prior_study': item['prior_study']
            })
        # self.examples = self.examples[:10]

    def __len__(self):
        return len(self.examples)


class MimiccxrFinetuneDataset(FinetuneBaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        sample = (example['id'], example['anchor_scan'], example['auxiliary_references'],
                  example['report'], example['indication'], example['prior_study'])
        return sample


class PretrainDinov2CollateFn:
    def __init__(self, image_dir, processor, is_multiview_learning, is_prior_scan):
        self.image_dir = image_dir
        self.processor = processor
        self.is_multiview_learning = is_multiview_learning
        self.is_prior_scan = is_prior_scan

    def __call__(self, data):
        # note that images is tuple(list, list)
        image_ids, anchor_scans, auxiliary_scans, reports, prior_studies = zip(*data)

        # images
        images, view_positions, patient_ids, patient_info = [], [], [], []
        # patient_info is used to avoid adding redundant image
        # patient_ids is used to record the types of each radiographs
        # (including anchor scan, auxiliary references, and prior study)
        # 1. preprocess batch images
        for anchor_item in anchor_scans:
            image_path, vp = anchor_item['image_path'][0], anchor_item['view_position'][0]
            image_path_split = image_path.split('/')

            # add patient id and information
            cur_patient_id = '_'.join(image_path_split[1:3])
            cur_patient_info = '_'.join(image_path_split[1:])
            patient_info.append(cur_patient_info)
            patient_ids.append(cur_patient_id)

            # add image info
            ## global image patches
            image = Image.open(os.path.join(self.image_dir, image_path))
            image = self.processor(image, return_tensors='pt').pixel_values
            images.append(image)
            view_positions.append(vp)

        # 2. preprocess multiview images
        if self.is_multiview_learning:
            for aux_item in auxiliary_scans:
                multiview_images, multiview_vp = aux_item['image_path'], aux_item['view_position']
                for mv_image, mv_vp in zip(multiview_images, multiview_vp):
                    mv_image_path_split = mv_image.split('/')

                    cur_patient_info = '_'.join(mv_image_path_split[1:])
                    if cur_patient_info not in patient_info:
                        # add patient id and information
                        cur_patient_id = '_'.join(mv_image_path_split[1:3])
                        patient_info.append(cur_patient_info)
                        patient_ids.append(cur_patient_id)

                        # add image info
                        ## global image patches
                        image = Image.open(os.path.join(self.image_dir, mv_image))
                        image = self.processor(image, return_tensors='pt').pixel_values
                        images.append(image)
                        view_positions.append(mv_vp)

        # record the number of radiographs for each patient; determine current/prior images
        # 3. preprocess prior images
        if self.is_prior_scan:
            valid_prior_images = []  # record the path of prior_image, avoid repeat (i.e., radiographs from the same study)
            for p_study, p_patient_id in zip(prior_studies, patient_ids[:len(prior_studies)]):
                if p_study is None:
                    continue
                for k, v in p_study.items():
                    if k != 'latest_study':
                        continue
                    if v['image_path'] in valid_prior_images:
                        continue
                    valid_prior_images.append(v['image_path'])
                    # prior marks current/prior scans; k[0] marks the first/second most recent prior; view_position
                    view_positions.append(f'{v["view_position"]}_{k.split("_")[0]}_prior')

                    # add image info
                    ## global image patches
                    image = Image.open(os.path.join(self.image_dir, v['image_path']))
                    image = self.processor(image, return_tensors='pt').pixel_values
                    images.append(image)
                    patient_ids.append(p_patient_id)

        images = torch.cat(images, dim=0)  # (image_num, 3, 518, 518)
        patient_ids = np.array(patient_ids)
        view_positions = np.array(view_positions)
        # view_position record the view_position of each radiographs, and the prior images are "prior"
        return image_ids, images, list(reports), patient_ids, view_positions


class FinetuneDinov2CollateFn:
    def __init__(self, args, processor):
        self.processor = processor
        self.args = args

    def __call__(self, data):
        image_ids, anchor_scans, auxiliary_scans, reports, indications, prior_scans = zip(*data)
        # images
        images, view_positions, patient_ids, patient_info = [], [], [], []
        # patient_info is used to avoid adding redundant image
        # patient_ids is used to record the types of each radiographs
        # (including anchor scan, auxiliary references, and prior study)
        # 1. preprocess batch images
        for anchor_item in anchor_scans:
            image_path, vp = anchor_item['image_path'][0], anchor_item['view_position'][0]
            image_path_split = image_path.split('/')

            # add patient id and information
            cur_patient_id = '_'.join(image_path_split[1:3])
            cur_patient_info = '_'.join(image_path_split[1:])
            patient_info.append(cur_patient_info)
            patient_ids.append(cur_patient_id)

            # add image info
            ## global image patches
            image = Image.open(os.path.join(self.args['images_dir'], image_path))
            image = self.processor(image, return_tensors='pt').pixel_values
            images.append(image)
            view_positions.append(vp)

        # 2. preprocess multiview images
        if self.args['is_multiview_learning']:
            for aux_item in auxiliary_scans:
                multiview_images, multiview_vp = aux_item['image_path'], aux_item['view_position']
                for mv_image, mv_vp in zip(multiview_images, multiview_vp):
                    mv_image_path_split = mv_image.split('/')

                    cur_patient_info = '_'.join(mv_image_path_split[1:])
                    if cur_patient_info not in patient_info:
                        # add patient id and information
                        cur_patient_id = '_'.join(mv_image_path_split[1:3])
                        patient_info.append(cur_patient_info)
                        patient_ids.append(cur_patient_id)

                        # add image info
                        ## global image patches
                        image = Image.open(os.path.join(self.args['images_dir'], mv_image))
                        image = self.processor(image, return_tensors='pt').pixel_values
                        images.append(image)
                        view_positions.append(mv_vp)

        # record the number of radiographs for each patient; determine current/prior images
        # 3. preprocess prior images
        if self.args['is_prior_scan']:
            valid_prior_images = []  # record the path of prior_image, avoid repeat (i.e., radiographs from the same study)
            for p_study, p_patient_id in zip(prior_scans, patient_ids[:len(prior_scans)]):
                if p_study is None:
                    continue
                for k, v in p_study.items():
                    if k != 'latest_study':
                        continue
                    if v['image_path'] in valid_prior_images:
                        continue
                    valid_prior_images.append(v['image_path'])
                    # =prior marks current/prior scans; k[0] marks the first/second most recent prior; view_position
                    view_positions.append(f'{v["view_position"]}_{k.split("_")[0]}_prior')

                    # add image info
                    ## global image patches
                    image = Image.open(os.path.join(self.args['images_dir'], v['image_path']))
                    image = self.processor(image, return_tensors='pt').pixel_values
                    images.append(image)
                    patient_ids.append(p_patient_id)

        prior_reports = []
        if self.args['is_prior_report']:
            for p_study, p_patient_id in zip(prior_scans, patient_ids[:len(prior_scans)]):
                report = '[NHPR][SEP]'
                if p_study:
                    cur_prior_report = []
                    for k, v in p_study.items():
                        if k != 'latest_study':
                            continue
                        if len(cur_prior_report) == 0 and v['findings'] is not None:
                            if self.args['report_style'] == 'factual_serialization':
                                findings_fs = list(map(lambda x: x.strip().lower(), v['findings_factual_serialization']))
                                item = "[SEP]".join(findings_fs) + '[SEP]'
                                cur_prior_report.append(item)
                            else:
                                cur_prior_report.append(v['findings'].strip().lower() + '[SEP]')
                    if len(cur_prior_report) != 0:
                        report = cur_prior_report[0]
                if not self.args['is_indication']:
                    report = '[CLS]' + report
                prior_reports.append(report)
        else:
            indications = [i+'[SEP]' for i in indications]
        images = torch.cat(images, dim=0)  # (image_num+1*batch_size, 5, 3, 518, 518)
        patient_ids = np.array(patient_ids)
        view_positions = np.array(view_positions)
        # view_position record the view_position of each radiographs, and the prior images are "prior"
        return image_ids, images, list(reports), patient_ids, view_positions, indications, prior_reports


import os
import json
import base64
import logging
import subprocess
import numpy as np
import pandas as pd
from pycocotools import mask as cocomask
from skimage.measure import find_contours, label
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from tqdm import tqdm

from pprint import pprint


class CocoDatasetHandler:
    def __init__(self, jsonpath, jsontemplate, imgpath):
        with open(jsonpath, 'r') as jsonfile:
            anns = json.load(jsonfile)

        with open(jsontemplate, 'r') as jsonfile:
            templates = json.load(jsonfile)

        images = pd.DataFrame.from_dict(templates['images']).set_index('id')
        categories = pd.DataFrame.from_dict(templates['categories']).set_index('id')
        annotations = pd.DataFrame(dict(category_id=[], image_id=[]), dtype=int)
        logging.info('Reading annotations...')
        for ann in tqdm(anns):
        # for ann in anns:
        #     if ann['image_id'] not in [447]:
        #         continue
            annotations = annotations.append(ann, ignore_index=True)

        logging.info('Grouping annotations by label...')
        annotations['segmentation'] = annotations.apply(self.add_score, axis=1)
        annotations = annotations.groupby(['image_id','category_id'])['segmentation'].apply(list).reset_index()
        annotations = annotations.merge(images, left_on='image_id', right_index=True)
        annotations = annotations.merge(categories, left_on='category_id', right_index=True)
        logging.info('Decoding segmentation RLE...')
        annotations = annotations.assign(
            shapes = annotations.apply(self.rle2shape, axis=1))

        self.labelme = {}
        self.annotations = annotations
        self.imgpath = imgpath
        self.images = pd.DataFrame.from_dict(templates['images']).set_index('file_name')

    def add_score(self, row):
        row['segmentation']['score'] = row['score']
        return row['segmentation']

    def rle2shape(self, row):
        scores = []
        mask_polygons = []
        img_area = None

        img_mask = None
        for rle in row['segmentation']:
            mask = cocomask.decode(rle)
            mask = label(mask)
            mask = mask == np.argmax(np.bincount(mask.flat)[1:])+1  # filter noise 
            mask_polygons.append(find_contours(self.create_padded_mask(mask), 0.8)[0])
            if img_mask is None:
                img_mask = mask
                if not img_area: img_area = img_mask.shape[0] * img_mask.shape[1]
            else:
                prev_mask_bin = np.bincount(img_mask.flat)[1]
                img_mask = img_mask | mask
        padded_mask = self.create_padded_mask(img_mask)
        padded_mask = ndi.binary_fill_holes(padded_mask)
        shapes = find_contours(padded_mask, 0.8)

        # if extracted contour is empty
        if not shapes:
            return shapes
        mask_polygons = [
            [[int(point[1]), int(point[0])] for point in polygon]
            for polygon in mask_polygons
        ]
        shapes = [
            [[int(point[1]), int(point[0])] for point in polygon]
            for polygon in shapes
        ]
        for polygon in shapes:
            p1 = Polygon(polygon)
            thr_list = [p1.intersection(Polygon(p2)).area / p1.area for p2 in mask_polygons]
            scores.append(row['segmentation'][thr_list.index(max(thr_list))]['score'])
        return scores, shapes

    def create_padded_mask(self, mask):
        padded_mask = np.zeros(
            (mask.shape[0]+2, mask.shape[1]+2),
            dtype=np.uint8,
        )
        padded_mask[1:-1, 1:-1] = mask
        return padded_mask

    def check_overlap(self, score1, polygon1, current_row, df, filename, thr=0.7, thr2=0.9):
        p1 = Polygon(polygon1)
        for (_, iter_row) in df.iterrows():
            if current_row['name'] == iter_row['name']:
                continue
            for score2, polygon2 in zip(iter_row.shapes[0], iter_row.shapes[1]):
                p2 = Polygon(polygon2)
                if p1.intersection(p2).area / p1.area >= thr and p2.intersection(p1).area / p2.area >= thr:
                    if current_row['name'] == 'LDK' and iter_row['name'] == '洋室' and score2 - score1 > 0.2:
                        return False
                    if current_row['name'] == '廊下' and iter_row['name'] == '洋室' and score2 - score1 > 0.2:
                        return False
                # if current_row['name'] == '廊下' and iter_row['name'] == 'LDK':
                #     if score2 > score1 and p1.intersection(p2).area / p1.area >= thr2:
                #         return False
                # if current_row['name'] == 'LDK' and iter_row['name'] == '廊下':
                #     if score2 > score1 and p1.intersection(p2).area / p1.area >= thr2:
                #         return False
        return True

    def coco2labelme(self):
        fillColor = [255, 0, 0, 128]
        lineColor = [0, 255, 0, 128]

        groups = self.annotations.groupby('file_name')
        logging.info('Converting segmentation mask into polygons...')
        for file_idx, (filename, df) in tqdm(enumerate(groups)):
            record = {
                'imageData': base64.b64encode(open(os.path.join("test_template_annotations_coco", \
                    filename), "rb").read()).decode('utf-8'),
                'fillColor': fillColor,
                'lineColor': lineColor,
                'imagePath': filename,
                'imageHeight': int(self.images.loc[filename].height),
                'imageWidth': int(self.images.loc[filename].width),
            }
            record['shapes'] = []

            instance = {
                'line_color': None,
                'fill_color': None,
            }
            for inst_idx, (_, row) in enumerate(df.iterrows()):
                for score, polygon in zip(row.shapes[0], row.shapes[1]):
                    copy_instance = instance.copy()
                    # if extracted contour is empty
                    if not row.shapes:
                        continue
                    if not self.check_overlap(score, polygon, row, df, filename):
                        continue
                    copy_instance.update({
                        'shape_type': "polygon",
                        'label': row['name'],
                        'group_id': inst_idx,
                        'points': polygon
                    })
                    record['shapes'].append(copy_instance)

            if filename not in self.labelme.keys():
                self.labelme[filename] = record

    def save_labelme(self, file_names, dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        logging.info('Saving processed files...')
        for file in tqdm(file_names):
            filename = os.path.splitext(os.path.basename(file))[0]
            with open(os.path.join(dirpath, "{}.json".format(filename)), 'w', encoding="utf-8") as jsonfile:
                json.dump(self.labelme[file], jsonfile)


if __name__ == "__main__":
    ds = CocoDatasetHandler(
        'test_30thr_nodoor/mask_detections.json', 
        'test_template_annotations_coco/annotations.json', 
        'test_template_annotations_coco/JPEGImages'
        )
    ds.coco2labelme()
    ds.save_labelme(ds.labelme.keys(), 'test_annotations_labelme/')
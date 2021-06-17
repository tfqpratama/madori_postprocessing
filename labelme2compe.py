import glob
import json
import base64
import cv2
from tqdm import tqdm
from os import path as osp

from pprint import pprint


def main():
	rectangle_obj = ["開戸","折戸","引戸"]

	annot_files = glob.glob("test_annotations_labelme/*")
	compe = {}
	for annot_file in tqdm(annot_files):
		filename = osp.splitext(osp.basename(annot_file))[0]
		filedict = {}

		with open(annot_file, "r", encoding="utf-8") as f:
			annot = json.load(f)

		for shape in annot["shapes"]:
			# if label not exists previously
			if shape['label'] not in filedict and shape['label'] != '洋室':
				filedict[shape['label']] = []
			# insert each label to dictionary
			if shape['label'] in rectangle_obj:
				coor = [val for point in shape['points'] for val in point]
			elif shape['label'] != '洋室':
				coor = shape['points']
			else:
				continue
			filedict[shape['label']].append(coor)

		compe[filename] = filedict
		
	with open("mask_output.json", "w", encoding="utf-8") as f:
		json.dump(compe, f)

if __name__ == "__main__":
	main()
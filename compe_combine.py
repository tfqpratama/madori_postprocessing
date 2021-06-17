import argparse
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask-output', default = './mask_output.json')
    parser.add_argument('--bbox-output', default = './bbox_output.json')
    parser.add_argument('--combined-output', default = './madori_submission.json')

    args = parser.parse_args()

    return args

def main():
    # parse arguments
    args = parse_args()

    # load the prediction file
    with open(args.mask_output, encoding='utf-8') as f:
        mask = json.load(f)
    with open(args.bbox_output, encoding='utf-8') as f:
        bbox = json.load(f)

    combined = {}
    mask_label = ['LDK', '廊下', '浴室']
    bbox_label = ['開戸', '引戸', '折戸']

    for filename, _ in tqdm(mask.items()):
        shapes = {}
        for label in mask_label:
            if label in mask[filename]: shapes[label] = mask[filename][label]
        for label in bbox_label:
            if label in bbox[filename]: shapes[label] = bbox[filename][label]
        combined[filename] = shapes

    with open(args.combined_output, 'w', encoding='utf-8') as f:
        bbox = json.dump(combined, f)

if __name__ == "__main__":
    main()
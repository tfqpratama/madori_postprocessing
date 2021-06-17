#!/bin/sh
echo "Converting COCO to labelme format:"
python coco2labelme.py
echo "Converting labelme to competition format:"
python labelme2compe.py
echo "Combining object detection and segmentation:"
python compe_combine.py
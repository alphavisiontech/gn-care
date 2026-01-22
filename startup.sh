#!/bin/bash
source /home/orangepi/Desktop/anaconda3/bin/activate
conda activate fd

sleep 60s
python main_rknn.py --input-type webcam --score-threshold 0.6 --keypoint-threshold 0.3 --fall-consecutive-threshold 10 --ratio-threshold 1.2 --backward-ratio-threshold 0.6

shutdown -r 00:00
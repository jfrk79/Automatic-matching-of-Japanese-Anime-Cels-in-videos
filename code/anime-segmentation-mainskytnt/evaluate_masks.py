import os
import argparse
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_masks(gt_dir, pred_dir, threshold=0.5):
    # Debug: Print the directories being used
    print(f"Ground truth directory: {gt_dir}")
    print(f"Prediction directory: {pred_dir}")

    gt_paths = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.jpg')])
    pred_paths = sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith('.jpg')])

    # Debug: Print lengths of file lists and the first few file names
    print(f"Number of ground truth files: {len(gt_paths)}")
    print(f"Number of prediction files: {len(pred_paths)}")
    print("Ground truth files:", gt_paths[:5])
    print("Prediction files:", pred_paths[:5])

    assert len(gt_paths) == len(pred_paths), "Ground truth and prediction directories must contain the same number of files."

    precisions = []
    recalls = []
    f1s = []

    for gt_path, pred_path in zip(gt_paths, pred_paths):
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) / 255.0
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) / 255.0

        gt_mask = (gt_mask > threshold).astype(np.uint8)
        pred_mask = (pred_mask > threshold).astype(np.uint8)

        precision = precision_score(gt_mask.flatten(), pred_mask.flatten())
        recall = recall_score(gt_mask.flatten(), pred_mask.flatten())
        f1 = f1_score(gt_mask.flatten(), pred_mask.flatten())

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(f"File: {os.path.basename(gt_path)}")
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  F1 Score: {f1:.4f}')
        print("")

    print("Average Scores:")
    print(f'Precision: {np.mean(precisions):.4f}')
    print(f'Recall: {np.mean(recalls):.4f}')
    print(f'F1 Score: {np.mean(f1s):.4f}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation masks")
    parser.add_argument('--gt-dir', type=str, required=True, help='Directory containing ground truth masks')
    parser.add_argument('--pred-dir', type=str, required=True, help='Directory containing predicted masks')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold to binarize masks')
    opt = parser.parse_args()

    # Debug: Print the provided directory paths
    print(f"Provided ground truth directory: {opt.gt_dir}")
    print(f"Provided prediction directory: {opt.pred_dir}")

    evaluate_masks(opt.gt_dir, opt.pred_dir, opt.threshold)

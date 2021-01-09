import os
import copy
import time
import logging
import numpy as np

from config import cfg


def do_sk_evaluation(dataset, outputs, targets, file_paths, ignored_index=255):
    logger = logging.getLogger('eve.' + __name__)
    num_classes = dataset.num_classes
    total_seen = np.zeros(num_classes)
    total_correct = np.zeros(num_classes)
    total_positive = np.zeros(num_classes)
    total_seen_num = 0
    total_correct_num = 0

    for idx in outputs.keys():
        output = outputs[idx]
        target = targets[idx]

        output = output[target != ignored_index]
        target = target[target != ignored_index]

        for i in range(num_classes):
            total_seen[i] += np.sum(target == i)
            total_correct[i] += np.sum((target == i) & (output == target))
            total_positive[i] += np.sum(output == i)

        total_seen_num += target.size
        total_correct_num += np.sum(target == output)

    # cls accuracy
    cls_acc = np.zeros(num_classes)
    for i in range(len(cls_acc)):
        if total_seen[i] == 0:
            cls_acc[i] = 1.0
        else:
            cls_acc[i] = total_correct[i] / total_seen[i]
    cls_acc = np.mean(cls_acc)

    # iou
    ious = []
    for i in range(num_classes):
        if total_seen[i] == 0:
            ious.append(1)
        else:
            cur_iou = total_correct[i] / \
                (total_seen[i] + total_positive[i] - total_correct[i])
            ious.append(cur_iou)
    logger.info("IoUs {}".format(ious))
    iou = np.mean(ious)

    # overall acc
    overall_acc = total_correct_num / total_seen_num

    metrics = dict(overall_acc=overall_acc, cls_acc=cls_acc, iou=iou)

    if file_paths is not None:
        label_map_inv = dataset.label_map_inv
        save_dir = os.path.join(cfg.LOGS.DIR, 'sequences')
        save_seg_prediction(outputs, label_map_inv, file_paths, save_dir)

    return metrics


def save_seg_prediction(seg_predictions, label_map_inv, file_paths, folder_path):
    logger = logging.getLogger('eve.' + __name__)
    logger.info("Saving result to {}".format(folder_path))

    start_time = time.time()

    for i in seg_predictions.keys():
        pred_before_map = seg_predictions[i]
        pred = label_map_inv[pred_before_map]
        file_path = file_paths[i]

        file_name = file_path.split('sequences/')[-1]
        file_name = file_name.replace('velodyne', 'predictions')
        file_name = file_name.replace('.bin', '.label')
        file_path = os.path.join(folder_path, file_name)
        os.makedirs('/'.join(file_path.split('/')[:-1]), exist_ok=True)

        pred.astype(np.uint32).tofile(file_path)
    
    logger.info("Saving result costs {}s".format(time.time() - start_time))

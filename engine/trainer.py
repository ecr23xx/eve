import time
import logging
import datetime

import torch
import numpy as np
import MinkowskiEngine as ME
import torch.distributed as dist


from config import cfg
from utils.comm import synchronize, reduce_dict, is_main_process
from utils.metric_logger import MetricLogger
from engine.inference import inference


def do_train(cfg, model, data_loader, optimizer, scheduler,
             criterion, checkpointer, device, arguments,
             tblogger, data_loader_val, distributed):
    logger = logging.getLogger('eve.' + __name__)
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments['iteration']
    model.train()
    start_training_time = time.time()
    end = time.time()
    logger.info("Start training")
    logger.info("Arguments: {}".format(arguments))

    for iteration, batch in enumerate(data_loader, start_iter):
        model.train()
        data_time = time.time() - end
        iteration = iteration + 1
        arguments['iteration'] = iteration

        # FIXME: for eve, modify dataloader
        locs, feats, targets, _ = batch
        inputs = ME.SparseTensor(feats, coords=locs).to(device)
        targets = targets.to(device, non_blocking=True).long()
        out = model(inputs, y=targets)

        if len(out) == 2:  # minkunet_eve
            outputs, match = out
        else:
            outputs = out
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if len(out) == 2:  # FIXME
            loss_dict = dict(loss=loss, match_acc=match[0], match_time=match[1])
        else:
            loss_dict = dict(loss=loss)
        loss_dict_reduced = reduce_dict(loss_dict)
        meters.update(**loss_dict_reduced)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data_time=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if tblogger is not None:
            for name, meter in meters.meters.items():
                if 'time' in name:
                    tblogger.add_scalar(
                        'other/' + name, meter.median, iteration)
                else:
                    tblogger.add_scalar(
                        'train/' + name, meter.median, iteration)
            tblogger.add_scalar(
                'other/lr', optimizer.param_groups[0]['lr'], iteration)

        if iteration % cfg.SOLVER.LOG_PERIOD == 0 \
                or iteration == max_iter \
                or iteration == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        "train eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        scheduler.step()

        if iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpointer.save('model_{:06d}'.format(iteration), **arguments)

        if iteration % 100 == 0:
            checkpointer.save('model_last', **arguments)

        if iteration == max_iter:
            checkpointer.save('model_final', **arguments)

        if iteration % cfg.SOLVER.EVAL_PERIOD == 0 \
                or iteration == max_iter:
            metrics = val_in_train(
                model,
                criterion,
                cfg.DATASETS.VAL,
                data_loader_val,
                tblogger,
                iteration,
                checkpointer,
                distributed)

            if metrics is not None:
                if arguments['best_iou'] < metrics['iou']:
                    arguments['best_iou'] = metrics['iou']
                    logger.info('best_iou: {}'.format(arguments['best_iou']))
                    checkpointer.save('model_best', **arguments)
                else:
                    logger.info('best_iou: {}'.format(arguments['best_iou']))

            if tblogger is not None:
                tblogger.add_scalar(
                    'val/best_iou', arguments['best_iou'], iteration)

            model.train()

            end = time.time()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def val_in_train(model, criterion, dataset_name_val, data_loader_val,
                 tblogger, iteration, checkpointer, distributed):
    logger = logging.getLogger('eve.' + __name__)

    if distributed:
        model_val = model.module
    else:
        model_val = model

    # only main process will return result
    metrics = inference(model_val, criterion,
                        data_loader_val, dataset_name_val)

    synchronize()

    if is_main_process():
        if tblogger is not None:
            for k, v in metrics.items():
                tblogger.add_scalar('val/' + k, v, iteration)
                logger.info("{}: {}".format(k, v))
        return metrics
    else:
        return None

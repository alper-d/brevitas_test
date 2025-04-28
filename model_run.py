import os

from dataloader import train_loader, test_loader
import torch
import time

# import qonnx.core.onnx_exec as oxe
from imports import (
    log_to_file,
    start_log_to_file,
    save_best_checkpoint,
    export_best_onnx,
    TrainingEpochMeters,
    OneShotPruning,
    EvalEpochMeters,
    eval_model,
)
from models_folder.models import model_with_cfg
from qonnx.core.modelwrapper import ModelWrapper

# import onnx.numpy_helper as numpy_helper
# from onnx2torch import convert
from configurations import (
    cmd_args,
    pruning_amount,
    get_optimizer,
    eta_min,
    device,
    log_freq,
    path_for_save,
    pruning_type,
    now_str,
    T_max,
    T_mult,
    SqrHingeLoss,
    epochs,
    lr_schedule_period,
    num_classes,
    starting_epoch,
    lr_schedule_ratio,
    get_scheduler,
)
from shutil import make_archive

run_netron, pruning_mode, use_scheduler, model_identity, is_iterative, pretrained = (
    cmd_args["run_netron"],
    cmd_args["pruning_mode"],
    cmd_args["use_scheduler"],
    cmd_args["model_identity"],
    cmd_args["is_iterative"],
    cmd_args["pretrained"],
)


def prune_and_train():
    # model_blueprint = load_model("runs/SIMD_0.9_30_Mar_2025__19_10_52/extended_model_0_9_SIMD.onnx")
    build_dir = "models_folder"
    best_val_acc = 0
    export_onnx_path = build_dir + "/end2end_cnv_w1a1_export_to_download.onnx"
    export_onnx_path2 = build_dir + "/checkpoint.tar"
    model_temp = ModelWrapper(export_onnx_path)
    # model_temp2 = get_test_model_trained("CNV", 1, 1)
    model, _ = model_with_cfg(model_identity, pretrained=pretrained)
    # if os.path.exists(f"runs/{pruning_log_identity}/best_checkpoint.tar"):
    #    model_dict = torch.load(f"runs/{pruning_log_identity}/best_checkpoint.tar")
    #    model.load_state_dict(model_dict["state_dict"])
    #    starting_epoch = model_dict["epoch"] - 1
    #    optimizer = model_dict["optim_dict"]
    #    best_val_acc = model_dict["best_val_acc"]
    file1 = start_log_to_file(path_for_save)

    # actual model load

    # package = torch.load(export_onnx_path2, map_location="cpu")
    # model_state_dict = package["state_dict"]
    # model.load_state_dict(model_state_dict)
    pruner = OneShotPruning()
    model = pruner.prune_wrapper(
        model, pruning_amount, pruning_mode, run_netron, path_for_save
    )
    pruner.weight_histograms(model, path_for_save)
    criterion, optimizer = get_optimizer(model)
    # model = CNV(10, WEIGHT_BIT_WIDTH, ACT_BIT_WIDTH, 8, 3).to(device=device)

    eval_meters = EvalEpochMeters()
    scheduler = (
        get_scheduler(optimizer=optimizer, T_max=T_max, eta_min=eta_min, T_mult=T_mult)
        if use_scheduler
        else None
    )
    model.to(device)
    for epoch in range(starting_epoch, epochs):
        # Set to training mode
        model.train()
        criterion.train()

        epoch_meters = TrainingEpochMeters()
        start_data_loading = time.time()

        for i, data in enumerate(train_loader):
            input, target = data
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # for hingeloss only
            if isinstance(criterion, SqrHingeLoss):
                target = target.unsqueeze(1)
                target_onehot = torch.Tensor(target.size(0), num_classes).to(
                    device, non_blocking=True
                )
                target_onehot.fill_(-1)
                target_onehot.scatter_(1, target, 1)
                target = target.squeeze()
                target_var = target_onehot
            else:
                target_var = target

            # measure data loading time
            epoch_meters.data_time.update(time.time() - start_data_loading)

            # Training batch starts
            start_batch = time.time()
            output = model(input)
            loss = criterion(output, target_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(epoch + i / iters)
            # log_str = f"Scheduler step. Next epoch(s) run with lr={scheduler.get_last_lr()}"
            # log_to_file(file1, log_str)
            model.clip_weights(-1, 1)

            epoch_meters.batch_time.update(time.time() - start_batch)
            pred = output.data.argmax(1, keepdim=True)
            correct = pred.eq(target.data.view_as(pred)).sum()
            prec1 = 100.0 * correct.float() / input.size(0)
            if i % int(log_freq) == 0:
                log_to_file(file1, f"Epoch: {epoch} Batch: {i} accuracy {str(prec1)}")

            eval_meters.top1.update(prec1.item(), input.size(0))
        log_str = "LR no update"
        if scheduler and epoch <= 348:
            # scheduler.step(epoch + 1)
            scheduler.step()
            log_str = (
                f"Scheduler step. Next epoch(s) run with lr={scheduler.get_last_lr()}"
            )
        elif (epoch + 1) % lr_schedule_period == 0:
            optimizer.param_groups[0]["lr"] *= lr_schedule_ratio
            log_str = f"Next epoch(s) run with lr={optimizer.param_groups[0]['lr']}"
        log_to_file(file1, log_str)
        # Perform eval
        with torch.no_grad():
            top1avg, save_data_list = eval_model(
                model, criterion, test_loader, num_classes, epoch, device
            )
        log_to_file(
            file1,
            f"Epoch {epoch} complete. Train  accuracy {str(eval_meters.top1.avg)}",
        )
        log_to_file(file1, f"Epoch {epoch} complete. Test accuracy {str(top1avg)}")
        if top1avg >= best_val_acc:
            best_val_acc = top1avg
            best_path = os.path.join(f"{path_for_save}", "best_checkpoint.tar")
            save_best_checkpoint(model, optimizer, epoch, best_val_acc, best_path)
        else:
            pass
    log_to_file(file1, f"Training complete. Best val acc= {best_val_acc}")
    # Define input shape
    example_inputs = torch.randn(1, 3, 32, 32)

    # Export to QONNX format
    if os.path.exists(f"{path_for_save}/best_checkpoint.tar"):
        model_dict = torch.load(
            f"{path_for_save}/best_checkpoint.tar", map_location=torch.device("cpu")
        )
        model.load_state_dict(model_dict["state_dict"])
    export_best_onnx(
        model.to("cpu"),
        example_inputs=example_inputs,
        export_path=f"{path_for_save}/best_model_qonnx.onnx",
    )
    file1.close()
    make_archive(f"run_zip/{pruning_type}_{now_str}", "zip", path_for_save)


if __name__ == "__main__":
    prune_and_train()

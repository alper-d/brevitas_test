import os

from dataloader import train_loader, test_loader
import torch
import time
from brevitas.export import export_qonnx

# import qonnx.core.onnx_exec as oxe
from imports import (
    log_to_file,
    save_best_checkpoint,
    export_best_onnx,
    TrainingEpochMeters,
    SqrHingeLoss,
    prune_wrapper,
    EvalEpochMeters,
    eval_model,
)
from models_folder.models.CNV import cnv
import datetime
from qonnx.core.modelwrapper import ModelWrapper
from imports import get_test_model_trained
from load_model import load_model

# import onnx.numpy_helper as numpy_helper
# from onnx2torch import convert
from configurations import (
    run_netron,
    pruning_mode,
    pruning_amount,
    get_optimizer,
    device,
    log_freq,
)

# model_blueprint = load_model("runs/SIMD_0.9_30_Mar_2025__19_10_52/extended_model_0_9_SIMD.onnx")
build_dir = "models_folder"
export_onnx_path = build_dir + "/end2end_cnv_w1a1_export_to_download.onnx"
export_onnx_path2 = build_dir + "/checkpoint.tar"
model_temp = ModelWrapper(export_onnx_path)
model_temp2 = get_test_model_trained("CNV", 1, 1)
model = cnv("cnv_1w1a")
criterion, optimizer = get_optimizer(model)

now_time = datetime.datetime.now()

epoch_data = {"train": {}, "test": {}}
epoch_data["train"][str(pruning_amount)] = []
epoch_data["test"][str(pruning_amount)] = []
num_classes = 10
starting_epoch = 0
best_val_acc = 0
epochs = 30

pruning_log_identity = (
    f"{pruning_mode}_{str(pruning_amount)}_{now_time.strftime('%d_%b_%Y__%H_%M_%S')}"
)

# if os.path.exists(f"runs/{pruning_log_identity}/best_checkpoint.tar"):
#    model_dict = torch.load(f"runs/{pruning_log_identity}/best_checkpoint.tar")
#    model.load_state_dict(model_dict["state_dict"])
#    starting_epoch = model_dict["epoch"] - 1
#    optimizer = model_dict["optim_dict"]
#    best_val_acc = model_dict["best_val_acc"]
os.mkdir(f"runs/{pruning_log_identity}")
file1 = open(
    f"runs/{pruning_log_identity}/pruning_logs_{pruning_log_identity}.txt",
    "a",
)
log_to_file(
    file1,
    f"Starting to write at {now_time.strftime('%H:%M:%S%p on %d %B %Y')}\nPruning Amount: {pruning_amount}\nPruning Mode: {pruning_mode}\n\n",
)

# actual model load

package = torch.load(export_onnx_path2, map_location="cpu")
model_state_dict = package["state_dict"]
model.load_state_dict(model_state_dict)
model = prune_wrapper(
    model, pruning_amount, pruning_mode, run_netron, pruning_log_identity
)
# model = CNV(10, WEIGHT_BIT_WIDTH, ACT_BIT_WIDTH, 8, 3).to(device=device)

eval_meters = EvalEpochMeters()

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

        model.clip_weights(-1, 1)

        # measure elapsed time
        epoch_meters.batch_time.update(time.time() - start_batch)
        pred = output.data.argmax(1, keepdim=True)
        correct = pred.eq(target.data.view_as(pred)).sum()
        prec1 = 100.0 * correct.float() / input.size(0)
        if i % int(log_freq) == 0:
            log_to_file(file1, f"Epoch: {epoch} Batch: {i} accuracy {str(prec1)}\n")

        eval_meters.top1.update(prec1.item(), input.size(0))

    # Perform eval
    with torch.no_grad():
        top1avg, save_data_list = eval_model(
            model, criterion, test_loader, num_classes, epoch, device
        )
    # epoch_data["test"][str(pruning_amount)].append(top1avg)
    # epoch_data["train"][str(pruning_amount)].append(eval_meters.top1.avg)
    log_to_file(
        file1, f"Epoch {epoch} complete. Train  accuracy {str(eval_meters.top1.avg)}\n"
    )
    log_to_file(file1, f"Epoch {epoch} complete. Test accuracy {str(top1avg)}\n")
    if top1avg >= best_val_acc:
        best_val_acc = top1avg
        best_path = os.path.join(f"runs/{pruning_log_identity}", "best_checkpoint.tar")
        save_best_checkpoint(model, optimizer, epoch, best_val_acc, best_path)
    else:
        pass
        # checkpoint_best(epoch, f"checkpoint_pruning_amount-{pruning_amount:.3f}.tar")
log_to_file(file1, f"Training complete. Best val acc= {best_val_acc}")
# Define input shape
example_inputs = torch.randn(1, 3, 32, 32)

# Export to QONNX format
if os.path.exists(f"runs/{pruning_log_identity}/best_checkpoint.tar"):
    model_dict = torch.load(f"runs/{pruning_log_identity}/best_checkpoint.tar")
    model.load_state_dict(model_dict["state_dict"])
export_best_onnx(
    model,
    example_inputs,
    f"runs/{pruning_log_identity}/best_model_qonnx.onnx",
)
file1.close()

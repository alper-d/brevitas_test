import os

from dataloader import train_loader, test_loader
import torch
import time

# import qonnx.core.onnx_exec as oxe
from imports import (
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

build_dir = "models_folder"
epoch_data = {"train": {}, "test": {}}
epoch_data["train"][str(pruning_amount)] = []
epoch_data["test"][str(pruning_amount)] = []
num_classes = 10

file1 = open(
    os.path.join("runs", f"pruning_logs_{str(pruning_amount)}_{pruning_mode}.txt"), "a"
)
now = datetime.datetime.now().strftime("%H:%M%p on %d %B %Y")
file1.write(
    f"Starting to write at {now}\nPruning Amount: {pruning_amount}\nPruning Mode:{pruning_mode}"
)
file1.flush()


export_onnx_path = build_dir + "/end2end_cnv_w1a1_export_to_download.onnx"
export_onnx_path2 = build_dir + "/checkpoint.tar"
model_temp = ModelWrapper(export_onnx_path)
model_temp2 = get_test_model_trained("CNV", 1, 1)
# actual model load
model = cnv("cnv_1w1a")
package = torch.load(export_onnx_path2, map_location="cpu")
model_state_dict = package["state_dict"]
model.load_state_dict(model_state_dict)
model = prune_wrapper(model, pruning_amount, pruning_mode, run_netron)
# model = CNV(10, WEIGHT_BIT_WIDTH, ACT_BIT_WIDTH, 8, 3).to(device=device)

eval_meters = EvalEpochMeters()
criterion, optimizer = get_optimizer(model)
starting_epoch = 1
best_val_acc = 0
epochs = 30
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
            file1.write(f"Epoch: {epoch} Batch: {i} accuracy {str(prec1)}\n")
            file1.flush()

        eval_meters.top1.update(prec1.item(), input.size(0))

    # Perform eval
    with torch.no_grad():
        top1avg, save_data_list = eval_model(
            model, criterion, test_loader, num_classes, epoch, device
        )
    # epoch_data["test"][str(pruning_amount)].append(top1avg)
    # epoch_data["train"][str(pruning_amount)].append(eval_meters.top1.avg)
    file1.write(
        f"Epoch {epoch} complete. Train  accuracy {str(eval_meters.top1.avg)}\n"
    )
    file1.flush()
    file1.write(f"Epoch {epoch} complete. Test accuracy {str(top1avg)}\n")
    file1.flush()
    if top1avg >= best_val_acc:
        best_val_acc = top1avg
        # checkpoint_best(epoch, f"best_pruning_amount-{pruning_amount:.3f}.tar")
    else:
        pass
        # checkpoint_best(epoch, f"checkpoint_pruning_amount-{pruning_amount:.3f}.tar")
file1.close()

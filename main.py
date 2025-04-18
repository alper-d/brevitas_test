# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# from finn.util.basic import make_build_dir
# from finn.util.visualization import showInNetron
from qonnx.core.modelwrapper import ModelWrapper
import os
import torch
import onnx.numpy_helper as numpy_helper
from models_folder.models import model_with_cfg
# build_dir = os.environ["FINN_BUILD_DIR"]
from configurations_delete_later import (
    run_netron,
    model_identity,
    pruning_mode,
    pruning_amount,
    path_for_save
)
from brevitas.nn import QuantConv2d
from imports_delete_later import prune_wrapper, export_best_onnx
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f"Hi, {name}")  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    model, _ = model_with_cfg(model_identity, pretrained=True)
    example_inputs = torch.randn(1, 3, 32, 32)
    #model = prune_wrapper(model, pruning_amount, pruning_mode, run_netron, path_for_save)
    #if os.path.exists(f"{path_for_save}/best_checkpoint.tar"):
    #    model_dict = torch.load(f"{path_for_save}/best_checkpoint.tar", map_location=torch.device('cpu'))
    #    model.load_state_dict(model_dict["state_dict"])
    #export_best_onnx(
    #    model.to("cpu"),
    #    example_inputs=example_inputs,
    #    export_path=f"{path_for_save}/best_model_qonnx.onnx",
    #)

    for layer in model.conv_features:
        if isinstance(layer, QuantConv2d):
            print(layer)

    exit()
    path = "untouched_models_folder/end2end_cnv_w1a1_export_to_download.onnx"
    model = ModelWrapper(path)
    x = model.graph.initializer
    # showInNetron(path)
    model = get_test_model_trained("CNV", 1, 1)
    prune_brevitas_model(
        model,
    )
    for i, node in enumerate(model.graph.node):
        if not node.op_type == "Conv":
            continue
        current_node = model.graph.node[i]
        print(current_node)
        model.get_initializer(current_node)
    for initializer in x:
        W = numpy_helper.to_array(initializer)
    print_hi("PyCharm")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

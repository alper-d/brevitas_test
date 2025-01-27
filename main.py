# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# from finn.util.basic import make_build_dir
# from finn.util.visualization import showInNetron
import os
from qonnx.core.modelwrapper import ModelWrapper
import netron
from imports import get_test_model_trained, prune_brevitas_model
from IPython.display import IFrame
import onnx.numpy_helper as numpy_helper
from onnx2torch import convert

# build_dir = os.environ["FINN_BUILD_DIR"]


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f"Hi, {name}")  # Press ⌘F8 to toggle the breakpoint.


def showInNetron(model_filename: str, localhost_url: str = None, port: int = None):
    """Shows a ONNX model file in the Jupyter Notebook using Netron.

    :param model_filename: The path to the ONNX model file.
    :type model_filename: str

    :param localhost_url: The IP address used by the Jupyter IFrame to show the model.
     Defaults to localhost.
    :type localhost_url: str, optional

    :param port: The port number used by Netron and the Jupyter IFrame to show
     the ONNX model.  Defaults to 8081.
    :type port: int, optional

    :return: The IFrame displaying the ONNX model.
    :rtype: IPython.lib.display.IFrame
    """
    try:
        port = port or int(os.getenv("NETRON_PORT", default="8081"))
    except ValueError:
        port = 8081
    localhost_url = localhost_url or os.getenv("LOCALHOST_URL", default="localhost")
    netron.start(model_filename, address=("0.0.0.0", port), browse=False)
    return IFrame(src=f"http://{localhost_url}:{port}/", width="100%", height=400)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    path = "untouched_models_folder/end2end_cnv_w1a1_export_to_download.onnx"
    model = ModelWrapper(path)
    x = model.graph.initializer
    showInNetron(path)
    model = get_test_model_trained("CNV", 1, 1)
    prune_brevitas_model(model, )
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

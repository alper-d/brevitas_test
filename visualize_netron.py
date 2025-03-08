import netron
from IPython.display import IFrame
import os

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
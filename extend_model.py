#import copy
#
#import numpy as np
#import onnx
#
#
#def add_batch_nomalization(input="Conv_5_out0"):
#    pass
#
#
#def x(input="Conv_5_out0"):
#    conv1_output_node_name = "Conv1_Y"
#    # Dummy weights for conv.
#    conv1_in_channels = 3
#    conv1_out_channels = 32
#    conv1_kernel_shape = (3, 3)
#    conv1_pads = (1, 1, 1, 1)
#    conv1_W = np.ones(
#        shape=(conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)
#    ).astype(np.float32)
#    conv1_B = np.ones(shape=(conv1_out_channels)).astype(np.float32)
#    # Create the initializer tensor for the weights.
#    conv1_W_initializer_tensor_name = "Conv1_W"
#    conv1_W_initializer_tensor = create_initializer_tensor(
#        name=conv1_W_initializer_tensor_name,
#        tensor_array=conv1_W,
#        data_type=onnx.TensorProto.FLOAT,
#    )
#    conv1_B_initializer_tensor_name = "Conv1_B"
#    conv1_B_initializer_tensor = create_initializer_tensor(
#        name=conv1_B_initializer_tensor_name,
#        tensor_array=conv1_B,
#        data_type=onnx.TensorProto.FLOAT,
#    )
#
#    conv1_node = onnx.helper.make_node(
#        name="Conv1",  # Name is optional.
#        op_type="Conv",
#        # Must follow the order of input and output definitions.
#        # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
#        inputs=[
#            model_input_name,
#            conv1_W_initializer_tensor_name,
#            conv1_B_initializer_tensor_name,
#        ],
#        outputs=[conv1_output_node_name],
#        # The following arguments are attributes.
#        kernel_shape=conv1_kernel_shape,
#        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
#        pads=conv1_pads,
#    )
#
#
#def identify_adder_nodes(model):
#    cloned_nodes = []
#    last_unit_nodes = [
#        "BatchNormalization_4",
#        "BipolarQuant_5",
#        "BipolarQuant_13",
#        "Conv_5",
#    ]
#    for node in model.graph.node:
#        if node.name in last_unit_nodes:
#            cloned_nodes.append(node)
#    cloned_nodes = list(map(copy.deepcopy, cloned_nodes))
#    for node in cloned_nodes:
#        if node.name == "BatchNormalization":
#            pass
#    return

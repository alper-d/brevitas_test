import json
from imports import conv_layer_traverse, prune_brevitas_model, prune_brevitas_modelSIMD
import brevitas


def get_pruned_blueprint(model, json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    for layer_idx, layer in enumerate(conv_layer_traverse(model)):
        if layer_idx == 0:
            continue
        print(layer_idx)
        for layer_dict in data:
            if layer_idx == layer_dict["pruned_layer_index"]:
                pruning_mode = layer_dict["pruning_mode"]
                if pruning_mode == "structured":
                    prune_brevitas_model(
                        model,
                        layer,
                        NumColPruned=(
                            layer.in_channels
                            - layer_dict["pruning_entities"]["in_channels_new"]
                        ),
                    )
                else:
                    prune_brevitas_modelSIMD(
                        model,
                        layer,
                        SIMD_in=layer_dict["pruning_entities"]["SIMD_in"],
                        NumColPruned=(
                            layer.in_channels
                            - layer_dict["pruning_entities"]["in_channels_new"]
                        ),
                    )
    return model


if __name__ == "__main__":
    from models_folder.models import model_with_cfg

    model, _ = model_with_cfg("cnv_1w1a", pretrained=True)
    get_pruned_blueprint(
        model, "runs/SIMD_cnv_1w1a/21_Apr_2025__14_34_44/extended_model_pruned.json"
    )

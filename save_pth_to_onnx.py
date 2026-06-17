import argparse
import importlib
from pathlib import Path


MODEL_TYPES = {
    "small_classifier": "BoardInputSmallClassifier",
    "big_classifier": "BoardInputBigClassifier",
    "bigbig_classifier": "BoardInputBigBigClassifier",
    "regression": "BoardInputRegression",
}

CLASSIFIER_TYPES = {
    "small_classifier",
    "big_classifier",
    "bigbig_classifier",
}


def load_state_dict(pth_path):
    import torch

    checkpoint = torch.load(pth_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"{pth_path} does not contain a PyTorch state_dict.")

    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def infer_input_dim(state_dict):
    try:
        return state_dict["linear1.weight"].shape[1]
    except KeyError as error:
        raise KeyError("Could not infer input_dim: missing linear1.weight.") from error


def infer_output_dim(state_dict):
    output_layers = ("linear5.weight", "linear4.weight")

    for layer_name in output_layers:
        if layer_name in state_dict:
            return state_dict[layer_name].shape[0]

    raise KeyError("Could not infer output_dim: missing final linear layer weight.")


def build_model(model_type, input_dim, output_dim):
    project_models = importlib.import_module("models")
    model_class = getattr(project_models, MODEL_TYPES[model_type])
    return model_class(input_dim=input_dim, output_dim=output_dim)


def wrap_classifier_for_class_output(model):
    import torch

    class ClassOutputWrapper(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, x):
            logits = self.classifier(x)
            return torch.argmax(logits, dim=1)

    return ClassOutputWrapper(model)


def get_default_output_path(pth_path, model_type, classifier_output):
    if model_type in CLASSIFIER_TYPES and classifier_output == "class":
        return pth_path.with_name(f"{pth_path.stem}_class.onnx")

    return pth_path.with_suffix(".onnx")


def export_to_onnx(model, output_path, input_dim, batch_size, opset_version, output_name):
    import torch

    dummy_input = torch.randn(batch_size, input_dim, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["board_input"],
        output_names=[output_name],
        dynamic_axes={
            "board_input": {0: "batch_size"},
            output_name: {0: "batch_size"},
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert one of this project's .pth model state_dict files to ONNX."
    )
    parser.add_argument(
        "--pth",
        required=True,
        type=Path,
        help="Path to the .pth file containing the model weights.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_TYPES.keys(),
        help="Model architecture that matches the weights.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        help="Output .onnx path. Defaults to the .pth path with an .onnx extension.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        help="Input feature count. Inferred from linear1.weight when omitted.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        help="Output feature count. Inferred from the final linear layer when omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used during export. The exported ONNX batch axis is dynamic.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--classifier-output",
        choices=("class", "logits"),
        default="class",
        help=(
            "For classifier models, export either one predicted class index per input "
            "or the raw class logits."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if not args.pth.exists():
            raise FileNotFoundError(f"Could not find .pth file: {args.pth}")

        output_path = args.onnx or get_default_output_path(
            args.pth,
            args.model,
            args.classifier_output,
        )
        state_dict = load_state_dict(args.pth)
        input_dim = args.input_dim or infer_input_dim(state_dict)
        output_dim = args.output_dim or infer_output_dim(state_dict)

        model = build_model(args.model, input_dim, output_dim)
        model.load_state_dict(state_dict)
        model.eval()

        output_name = "prediction"
        if args.model in CLASSIFIER_TYPES and args.classifier_output == "class":
            model = wrap_classifier_for_class_output(model)
            model.eval()
            output_name = "predicted_class"

        export_to_onnx(
            model=model,
            output_path=output_path,
            input_dim=input_dim,
            batch_size=args.batch_size,
            opset_version=args.opset,
            output_name=output_name,
        )
    except ModuleNotFoundError as error:
        if error.name in {"torch", "onnx"}:
            raise SystemExit(
                f"Missing dependency: {error.name}. Install it in this Python environment "
                "before exporting the model."
            ) from error
        raise

    print(f"Saved ONNX model to {output_path}")
    print(f"Model: {args.model}, input_dim: {input_dim}, output_dim: {output_dim}")
    if args.model in CLASSIFIER_TYPES:
        print(f"Classifier output: {args.classifier_output}")


if __name__ == "__main__":
    main()

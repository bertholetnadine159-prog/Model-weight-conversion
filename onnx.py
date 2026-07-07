import argparse
import warnings
from pathlib import Path

import torch

from ultralytics import YOLO


class SegmentONNXWrapper(torch.nn.Module):
    """Flatten segment model outputs for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        preds, proto = self.model(images)
        if isinstance(preds, torch.Tensor):
            return preds, proto
        return tuple(preds) + (proto,)


def parse_imgsz(imgsz):
    if len(imgsz) == 1:
        return imgsz[0], imgsz[0]
    if len(imgsz) == 2:
        return imgsz[0], imgsz[1]
    raise ValueError("--imgsz expects one value, e.g. 640, or two values, e.g. 640 640")


def parse_args():
    parser = argparse.ArgumentParser(description="Export Segment_Efficient YAML model to ONNX opset 11.")
    parser.add_argument(
        "--model",
        default="runs/train/yolo11_LBL2/segment/yolo11-seg/weights/best.pt",
        help="Model YAML path or .pt weights path.",
    )
    parser.add_argument("--weights", default=None, help="Optional .pt weights path. Loaded directly by default.")
    parser.add_argument(
        "--load-into-yaml",
        action="store_true",
        help="Build --model YAML first, then load --weights into it. By default --weights is loaded directly.",
    )
    parser.add_argument("--output", default="./onnx/yolo11", help="Optional ONNX output path.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="Input image size: 640 or 640 640.")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size. RDK X5 PTQ requires static batch 1.")
    parser.add_argument("--device", default="cpu", help="Export device, e.g. cpu or cuda:0.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch/height/width axes.")
    return parser.parse_args()


def get_file_size(path):
    return f"{Path(path).stat().st_size / 1024 / 1024:.1f}MB"


def get_output_names(model):
    head = model.model[-1]
    head_name = type(head).__name__
    if head_name.startswith("Segment"):
        return [f"output_p{i + 3}" for i in range(head.nl)] + ["proto"]
    raise TypeError(f"当前脚本只支持 Segment 系列 head，当前 head 为: {head_name}")


def get_dynamic_axes(output_names):
    dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
    for name in output_names[:-1]:
        dynamic_axes[name] = {0: "batch", 1: f"{name}_height", 2: f"{name}_width"}
    dynamic_axes["proto"] = {0: "batch", 1: "mask_height", 2: "mask_width"}
    return dynamic_axes


def main():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()
    source = Path(args.weights or args.model)
    output = Path(args.output) if args.output else Path(f"{source.stem}_opset11.onnx")
    if output.suffix.lower() != ".onnx":
        output = output.with_suffix(".onnx")
    height, width = parse_imgsz(args.imgsz)
    device = torch.device(args.device)

    if args.weights and args.load_into_yaml:
        yolo = YOLO(str(args.model), task="segment")
        yolo.load(args.weights)
    else:
        yolo = YOLO(str(source), task="segment")

    print(f"Source model: {source} ({get_file_size(source)})")

    model = yolo.model.to(device).eval().float()
    for module in model.modules():
        if hasattr(module, "export"):
            module.export = True
            module.format = "onnx"
        if hasattr(module, "dynamic"):
            module.dynamic = args.dynamic

    dummy = torch.zeros(args.batch, 3, height, width, device=device)
    wrapper = SegmentONNXWrapper(model).to(device).eval()
    output_names = get_output_names(model)
    print(f"Detected head: {type(model.model[-1]).__name__}, ONNX outputs: {output_names}")

    dynamic_axes = get_dynamic_axes(output_names) if args.dynamic else None

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        opset_version=11,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    try:
        import onnx

        onnx_model = onnx.load(str(output))
        onnx.checker.check_model(onnx_model)
    except ImportError:
        print("ONNX export finished. Install onnx to run model checker.")

    print(f"Exported ONNX opset 11 model: {output} ({get_file_size(output)})")


if __name__ == "__main__":
    main()

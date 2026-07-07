# Model Weight Conversion

这是一个面向 YOLO11 分割模型的权重转换与 RDK/Horizon 端侧部署项目。仓库包含 ONNX 导出脚本、D-Robotics RDK 平台 BIN 模型推理脚本，以及一个已转换的 `YOLO11_LBL.bin` 模型文件。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `onnx.py` | 将 Ultralytics YOLO11 segmentation 模型导出为 ONNX opset 11。 |
| `Model deployment.py` | 在 RDK/Horizon 环境中加载 `.bin` 模型，执行海参分割推理、可视化、评估和测速。 |
| `YOLO11_LBL.bin` | 已转换的量化 BIN 模型，默认由部署脚本加载。 |
| `f990ae1251f042863e950c4d86e5f24b.png` | 项目示例图片资源。 |

## 环境依赖

ONNX 导出环境建议安装：

```bash
pip install torch ultralytics onnx
```

RDK/Horizon 部署环境需要：

```bash
pip install numpy opencv-python
```

同时需要目标设备已安装并可导入：

- `hobot_dnn`
- 可选：`horizon_nn.debug`，用于量化敏感度分析

## 导出 ONNX

默认从训练权重导出 ONNX：

```bash
python onnx.py --model runs/train/yolo11_LBL2/segment/yolo11-seg/weights/best.pt --output ./onnx/yolo11.onnx --imgsz 640
```

也可以指定 batch、设备和动态输入：

```bash
python onnx.py --model path/to/best.pt --output ./onnx/yolo11.onnx --imgsz 640 640 --batch 1 --device cpu
```

参数说明：

- `--model`: 模型 YAML 或 `.pt` 权重路径。
- `--weights`: 可选权重路径。
- `--load-into-yaml`: 先按 YAML 构建模型，再加载权重。
- `--output`: ONNX 输出路径。
- `--imgsz`: 输入尺寸，可传一个值或两个值。
- `--batch`: 导出 batch size，RDK X5 PTQ 通常使用静态 batch 1。
- `--dynamic`: 是否启用动态 batch、高度和宽度。

## RDK BIN 推理

单张图片或图片目录推理：

```bash
python "Model deployment.py" --model-path YOLO11_LBL.bin --test-img path/to/image_or_dir --img-save-path seg_results
```

使用默认数据集结构推理并计算 P、R、mAP50 和 FPS：

```bash
python "Model deployment.py" --model-path YOLO11_LBL.bin --data-dir valid --benchmark-bin
```

默认数据集结构：

```text
valid/
  images/
    xxx.jpg
  labels/
    xxx.txt
```

标签格式使用 YOLO segmentation 多边形格式：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

其中坐标为归一化坐标。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-path` | `YOLO11_LBL.bin` | BIN 模型路径。 |
| `--test-img` | `sea_cucumber` | 输入图片或图片目录。 |
| `--img-save-path` | `seg_results` | 推理结果保存路径。 |
| `--classes-num` | `1` | 类别数量。 |
| `--conf-thres` | `0.15` | 置信度阈值。 |
| `--iou-thres` | `0.45` | NMS IoU 阈值。 |
| `--mask-thres` | `0.5` | mask 二值化阈值。 |
| `--label-dir` | `None` | YOLO segmentation 标签目录。 |
| `--eval-iou-thres` | `0.5` | 计算 mAP50 时的 IoU 阈值。 |

## 计算量估算

如果需要从 ONNX 估算 MACs、GMACs 和 GFLOPs：

```bash
python "Model deployment.py" --estimate-compute --onnx-model ./onnx/yolo11.onnx
```

也可以通过 `model.yaml` 中的 `onnx_model` 字段自动读取 ONNX 路径：

```bash
python "Model deployment.py" --estimate-compute --model-yaml model.yaml
```

## 量化敏感度调试

在安装 Horizon 调试工具后可运行：

```bash
python "Model deployment.py" --debug-sensitivity --debug-model ./calibrated_model.onnx --debug-calibrated-data ./calibration_data/
```

## 输出说明

部署脚本会输出：

- 分割框、类别、置信度和 mask 面积日志。
- 带有分割 mask 和检测框的可视化图片。
- 可选的 P、R、mAP50 和 FPS 表格。
- 可选的 BIN 前向耗时和端到端耗时表格。

## 注意事项

- `Model deployment.py` 期望模型输出为 4 个张量：`p3`、`p4`、`p5` 和 `proto`。脚本也兼容部分拆分为 10 个输出张量的模型。
- 当前类别名为 `sea_cucumber`，如需多类别部署，请同步修改类别名称、`--classes-num` 和模型输出配置。
- RDK 推理输入会转换为 NV12，并自动进行 letterbox resize。

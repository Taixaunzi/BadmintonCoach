"""GPU检测和ONNX导出工具"""
import sys
from pathlib import Path


def check_gpu():
    """检测可用的GPU设备"""
    print("=== GPU 检测 ===\n")

    # PyTorch CUDA
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
        else:
            print("  CUDA: 不可用")

        if hasattr(torch.backends, "mps"):
            print(f"  MPS (Apple Silicon): {torch.backends.mps.is_available()}")
    except ImportError:
        print("PyTorch: 未安装")

    # ONNX Runtime
    print()
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__}")
        print(f"  Providers: {ort.get_available_providers()}")
        if "CUDAExecutionProvider" in ort.get_available_providers():
            print("  ✅ GPU推理可用 (CUDA)")
        elif "CoreMLExecutionProvider" in ort.get_available_providers():
            print("  ✅ GPU推理可用 (CoreML)")
        else:
            print("  ⚠️  仅CPU推理")
    except ImportError:
        print("ONNX Runtime: 未安装")

    # TensorRT
    print()
    try:
        import tensorrt
        print(f"TensorRT: {tensorrt.__version__}")
        print("  ✅ TensorRT加速可用")
    except ImportError:
        print("TensorRT: 未安装（可选，最快推理）")

    # 推荐配置
    print("\n=== 推荐配置 ===")
    try:
        import torch
        if torch.cuda.is_available():
            print("config.yaml:")
            print("  pose:")
            print("    backend: onnxruntime  # 或 tensorrt")
            print("    device: cuda")
            print("  ball:")
            print("    device: cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("config.yaml:")
            print("  pose:")
            print("    backend: onnxruntime")
            print("    device: mps")
            print("  ball:")
            print("    device: mps")
        else:
            print("config.yaml:")
            print("  pose:")
            print("    backend: onnxruntime")
            print("    device: cpu")
            print("  ball:")
            print("    device: cpu")
            print("\n提示: 安装CUDA版PyTorch可获得10-100倍加速")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("需要安装PyTorch")


def export_onnx():
    """导出TrackNet为ONNX格式（推理更快）"""
    print("=== 导出 TrackNet ONNX 模型 ===\n")

    model_path = Path("models/tracknet_best.pth")
    if not model_path.exists():
        print(f"错误: 模型文件不存在 {model_path}")
        print("请下载: https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl")
        return

    try:
        import torch
        from badmintoncoach.engine.tracknet_model import BallTrackerNet

        model = BallTrackerNet()
        model.load_state_dict(torch.load(str(model_path), map_location="cpu", weights_only=True))
        model.eval()

        onnx_path = model_path.with_suffix(".onnx")
        dummy = torch.randn(1, 9, 360, 640)

        print(f"导出中: {model_path} → {onnx_path}")
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=11,
        )
        print(f"✅ 导出成功: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print("\n使用ONNX推理:")
        print("  1. pip install onnxruntime  # 或 onnxruntime-gpu")
        print("  2. 系统会自动使用ONNX模型（比PyTorch快2-5倍）")

    except Exception as e:
        print(f"导出失败: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        export_onnx()
    else:
        check_gpu()

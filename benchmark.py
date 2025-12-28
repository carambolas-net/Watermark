import torch
import sys
import time

def get_benchmark_config(device, include_fp64=True):
    """根据设备能力自动配置测试项"""
    if device.type == 'cpu':
        configs = [
            ("FP32", torch.float32, 'highest', "GFLOPS"),
            ("BF16", torch.bfloat16, 'highest', "GFLOPS"), # 高端 CPU 支持 BF16 加速
        ]
        if include_fp64:
            configs.append(("FP64 (Double)", torch.float64, 'highest', "GFLOPS"))
        return configs, "N/A"

    # GPU 部分逻辑保持不变
    major, minor = torch.cuda.get_device_capability(device)
    cc = major + minor / 10
    configs = []
    if cc >= 8.9:
        configs.append(("FP8 (e4m3fn)", "fp8_special", 'high', "TFLOPS"))
    if cc >= 8.0:
        configs.append(("FP16", torch.float16, 'high', "TFLOPS"))
        configs.append(("BF16", torch.bfloat16, 'high', "TFLOPS"))
        configs.append(("TF32 (FP32 Mode)", torch.float32, 'high', "TFLOPS"))
    configs.append(("True FP32", torch.float32, 'highest', "TFLOPS"))
    if include_fp64:
        configs.append(("FP64 (Double)", torch.float64, 'highest', "TFLOPS"))
    return configs, cc

def run_benchmark(target_device="cuda:0", iterations=100, include_fp64=True):
    try:
        device = torch.device(target_device)
        torch.zeros(1).to(device)
    except Exception as e:
        print(f"无法访问设备 {target_device}: {e}")
        return

    configs, cc = get_benchmark_config(device, include_fp64)
    N = 4096 if device.type == 'cpu' else 8192 # CPU 测试规模略小，避免跑太久
    
    print(f"{'='*60}")
    print(f" 设备: {target_device} | 规模: {N}x{N} | 迭代: {iterations}")
    print(f"{'='*60}")

    for label, dtype_info, precision_mode, unit in configs:
        try:
            if device.type == 'cuda':
                torch.set_float32_matmul_precision(precision_mode)
            
            # 数据初始化
            if dtype_info == "fp8_special":
                a = torch.randn(N, N, device=device).to(torch.float8_e4m3fn)
                b = torch.randn(N, N, device=device).to(torch.float8_e4m3fn).t().contiguous().t()
                scale = torch.tensor([1.0], device=device, dtype=torch.float32)
                test_func = lambda: torch._scaled_mm(a, b, scale, scale, out_dtype=torch.float16)
            else:
                a = torch.randn(N, N, device=device, dtype=dtype_info)
                b = torch.randn(N, N, device=device, dtype=dtype_info)
                test_func = lambda: torch.matmul(a, b)

            # 预热
            for _ in range(5): test_func()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations): test_func()
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end) / iterations
            else:
                start_time = time.perf_counter()
                for _ in range(iterations): test_func()
                ms = ((time.perf_counter() - start_time) * 1000) / iterations

            # 算力计算
            # 1 TFLOPS = 1000 GFLOPS
            flops_val = (2 * (N**3)) / (ms / 1000)
            performance = flops_val / 1e12 if device.type == 'cuda' else flops_val / 1e9

            print(f"[{label:18}] -> 平均耗时: {ms:10.2f} ms | 算力: {performance:10.4f} {unit}")
            
            del a, b
            if device.type == 'cuda': torch.cuda.empty_cache()

        except Exception as e:
            print(f"[{label:18}] -> 跳过 ({e})")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    # 既然是高端 CPU，我们默认开启 FP64 测试
    run_benchmark(target_device=target, iterations=100, include_fp64=False)
"""Etapa 0 - Confere se a GPU está instalada e funcionando com o PyTorch.

Mostra versão do PyTorch/CUDA, nome e memória da placa e roda um teste rápido
de multiplicação de matrizes na GPU x CPU. Use a saída para o print do VSCode.

Uso:
    python src/verificar_gpu.py
"""
import platform
import time

import torch


def bench(dispositivo, n=2048, rep=10):
    a = torch.randn(n, n, device=dispositivo)
    b = torch.randn(n, n, device=dispositivo)
    (a @ b).sum().item()  # aquecimento
    t0 = time.time()
    for _ in range(rep):
        c = a @ b
    c.sum().item()  # força sincronizar
    seg = (time.time() - t0) / rep
    return 2 * n ** 3 / seg / 1e12  # TFLOPS


print(f"Python ........ {platform.python_version()}")
print(f"PyTorch ....... {torch.__version__}")
print(f"CUDA no torch . {torch.version.cuda}")

if torch.cuda.is_available():
    disp = "cuda"
    p = torch.cuda.get_device_properties(0)
    print("CUDA disponivel: SIM")
    print(f"GPU ........... {p.name}")
    print(f"VRAM .......... {p.total_memory / 1e9:.1f} GB")
    print(f"Compute cap. .. {p.major}.{p.minor}")
    print(f"cuDNN ......... {torch.backends.cudnn.version()}")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    disp = "xpu"
    print("CUDA disponivel: NAO (usando GPU Intel via XPU)")
    print(f"GPU ........... {torch.xpu.get_device_name(0)}")
else:
    disp = "cpu"
    print("CUDA disponivel: NAO - nenhuma GPU encontrada, rodando na CPU")

print(f"\nTeste de matmul 2048x2048:")
print(f"  CPU ........ {bench('cpu'):.3f} TFLOPS")
if disp != "cpu":
    print(f"  {disp.upper():<4} ....... {bench(disp):.3f} TFLOPS")

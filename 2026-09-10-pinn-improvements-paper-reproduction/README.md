# PINN 改进方法论文复现（2026-09-10）

本目录承接前一天的 Burgers PINN baseline 与 limitations 实验，进入“针对失败原因提出改进方法”的阶段。

本次复现两种方法：

1. Residual-based Adaptive Refinement（RAR）：解决均匀采样遗漏陡峭区域的问题；
2. Gradient-based Adaptive Loss Weighting：解决 PDE、IC、BC 梯度尺度不平衡的问题。

## 核心结果

| 实验 | 原始 PINN | 改进后 | 改善幅度 |
|---|---:|---:|---:|
| Burgers RAR | 0.1769 | **0.0101** | 相对 L2 下降 94.3% |
| Helmholtz 自适应加权 | 0.1678 | **0.0194** | 相对 L2 下降 88.4% |
| Burgers 加权迁移 | 0.1101 | **0.0598** | 相对 L2 下降 45.7% |

### RAR 与原始 PINN

![RAR comparison](paper_reproductions/rar/rar_vs_original_pinn.png)

### 自适应加权与原始 PINN

![Adaptive weighting comparison](paper_reproductions/adaptive_weight/adaptive_weight_vs_original_pinn.png)

完整的论文参数、实现差异和原因分析见 [复现实验明细](paper_reproductions/README.md)。

## 运行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-paper.txt
python run_paper_rar.py
python run_paper_adaptive_weight.py
python make_paper_comparison_figures.py
```

前两个脚本完成训练并保存指标 checkpoint；最后一个脚本读取 checkpoint，生成原始 PINN 与改进方法的直接对比图。仓库只保留源码、汇总指标和关键图片，不提交大体积 checkpoint 与逐步训练日志。

# AI4PDE Experiments

AI4PDE / Scientific Machine Learning 学习实验与结果。

项目目录、实验 README、结果图、指标与 Git 上传统一遵循 [项目管理与上传规范](PROJECT_UPLOAD_GUIDE.md)。

## Experiments

- [2026-09-09 — Burgers Equation PINN and limitation studies](2026-09-09-burgers-pinn-limitations/README.md)
- [2026-09-10 — RAR and adaptive loss weighting reproductions](2026-09-10-pinn-improvements-paper-reproduction/README.md)
- [2026-09-14 — Cornell DeepONet nonlinear Darcy reproduction](2026-09-14-deeponet-nonlinear-darcy/README.md)

学习顺序为：PyTorch Burgers baseline → 黏性、采样与 loss imbalance 限制 → RAR 和梯度自适应损失加权 → DeepONet source-to-solution operator learning。每个日期目录包含可运行代码、关键指标、对照图片和实验说明。

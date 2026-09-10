# AI4PDE 项目管理与 GitHub 上传规范

版本：v1.0

生效日期：2026-09-10

本规范用于管理本仓库后续所有 AI4PDE / Scientific Machine Learning 学习实验。目标是让每次上传都能回答五个问题：

1. 为什么做这个实验？
2. 与前一个实验是什么关系？
3. 具体使用了什么配置？
4. 得到了什么结果？
5. 别人能否重新运行并验证？

## 1. 仓库整体结构

每个学习阶段使用一个独立的日期目录：

```text
AI4PDE_lllltd/
├── README.md
├── PROJECT_UPLOAD_GUIDE.md
├── YYYY-MM-DD-topic-name/
│   ├── README.md
│   ├── requirements.txt
│   ├── source_code.py
│   ├── results/
│   │   ├── metrics_summary.csv
│   │   ├── main_comparison.png
│   │   └── other_result.png
│   └── references/
│       └── README.md
└── YYYY-MM-DD-next-topic/
```

不要把新实验直接散落在仓库根目录。根目录只保留：

- 总目录 `README.md`；
- 本管理规范；
- 按日期组织的实验文件夹；
- 将来确实需要全仓库共享的配置文件。

## 2. 日期目录命名

统一格式：

```text
YYYY-MM-DD-short-topic-name
```

规则：

- 日期使用实验整理或上传日期；
- 主题使用小写英文和连字符；
- 名称应表达研究内容，不使用 `test1`、`new`、`final` 等无意义名称；
- 同一天存在多个独立主题时，在主题名中区分，不添加 `final-v2`。

示例：

```text
2026-09-09-burgers-pinn-limitations
2026-09-10-pinn-improvements-paper-reproduction
2026-09-15-deeponet-baseline
```

## 3. 每次必须上传的内容

每个日期目录至少包含以下内容。

### 3.1 README.md

README 是实验入口，必须包含：

1. 本实验在学习主线中的位置；
2. 前一种方法哪里不够好；
3. 新方法如何改进；
4. PDE、区域、初边值条件；
5. 网络、采样点、优化器、学习率、迭代数和随机种子；
6. 运行命令；
7. 核心结果表；
8. 关键图片及读图说明；
9. 与 baseline 或论文结果的差异；
10. 差异原因和下一步实验。

### 3.2 可运行源码

- 上传完成实验所需的 `.py` 文件；
- 主训练脚本使用 `run_*.py` 或 `train_*.py`；
- 绘图脚本可使用 `make_*_figures.py`；
- 公共函数放入名称明确的模块，例如 `burgers_reference.py`；
- 关键步骤写中文注释，但变量名和函数名使用清晰英文；
- 路径使用相对路径，不写死个人电脑绝对路径；
- 固定随机种子，并支持 CPU/CUDA 时明确说明。

### 3.3 依赖文件

至少提供一个 `requirements.txt`。如果论文复现需要额外依赖，可以增加：

```text
requirements-paper.txt
requirements-deeponet.txt
```

不要上传整个 `.venv`。

### 3.4 核心结果

至少上传：

- 一个汇总指标 CSV；
- 一张“原方法 vs 改进方法”的直接对照图；
- 对理解结论必要的其他图片；
- 如果结果失败，也要上传并解释，不得只保留成功实验。

## 4. 结果图片规范

图片必须让读者不看代码也能判断结果。

### 必须做到

- 文件名使用英文小写加下划线，例如 `rar_vs_original_pinn.png`；
- 分辨率建议不低于 160 DPI；
- 坐标轴、单位、时间和变量名称完整；
- 图例明确区分 `Exact`、`Original PINN` 和 `Improved method`；
- 对比热图必须使用相同色标；
- 误差图标明是 absolute error 还是 relative error；
- 柱状图注明“越低越好”或“越高越好”；
- 正文解释读者应该观察哪里。

### 推荐的对照图结构

```text
第一行：同一时间截面的 Exact / Original / Improved 曲线
第二行：Original error / Improved error / 指标柱状图
```

不要只上传单独一张改进后曲线，因为无法判断它是否真的优于 baseline。

## 5. 指标文件规范

统一使用 CSV，第一行为字段名。至少包含：

```text
method,relative_l2,pde_residual_rmse,ic_rmse,bc_rmse
```

论文复现可以增加：

```text
paper_reported_value,reproduction_value,seed,iterations,notes
```

规则：

- 保留足够的小数位，不只写图上的四舍五入值；
- README 中显示适合阅读的精简值；
- 明确指标是在训练点、固定验证点还是完整规则网格上计算；
- 不能只报告 total loss；
- PINN 至少同时检查解误差、PDE residual、IC 和 BC。

## 6. 论文复现证据等级

上传论文复现实验时，必须标明属于哪一种：

### A. 官方代码复现

公开代码和数据完整，主要改动只是框架兼容、结果保存或绘图。README 必须给出原仓库和论文链接。

### B. 论文配置移植

论文给出了网络与训练参数，但案例代码、几何数据或随机种子没有公开。必须明确写出缺失内容和自行实现部分，不能称为“完全复现”。

### C. 方法迁移实验

把论文算法应用到论文没有测试的新 PDE 或新数据上。必须与原论文实验分开报告，不能把迁移结果写成论文结果。

所有论文数值必须区分：

```text
论文报告值 / 本次复现值 / 本次迁移值
```

## 7. 不上传的内容

默认不上传：

```text
.venv/
__pycache__/
.DS_Store
.matplotlib-cache/
smoke-results/
临时下载文件
调试日志
重复图片
大型模型 checkpoint
每一步完整训练历史
```

模型文件只有在满足以下条件之一时才上传：

- 文件较小且对复现实验不可替代；
- 用户明确要求提供预训练模型；
- 已使用 Git LFS，并在 README 说明用途。

否则上传代码、随机种子、配置和指标，让读者自行训练。

## 8. 上传前检查清单

上传前依次检查：

- [ ] 日期目录名称符合规范；
- [ ] README 能说明“问题 → 方法 → 设置 → 结果 → 原因”；
- [ ] 运行命令可以从日期目录直接执行；
- [ ] 源码没有个人绝对路径；
- [ ] Python 文件通过语法检查；
- [ ] 至少完成一次快速运行或正式训练；
- [ ] 图片能够打开，没有裁切、重叠和色标歧义；
- [ ] 原始 PINN 与改进方法采用可解释的对照；
- [ ] 指标 CSV 与 README 数值一致；
- [ ] 论文、官方代码和第三方实现已明确区分；
- [ ] `.venv`、缓存、checkpoint 和临时文件没有进入提交；
- [ ] 检查 Git diff，确认没有删除或覆盖之前的实验；
- [ ] 拉取远程最新 `main` 后再推送。

## 9. Git 提交规范

一次提交只对应一个完整学习阶段或一次明确修正。

推荐提交信息：

```text
Add Burgers PINN limitation experiments
Add paper-style PINN improvement reproductions
Improve RAR comparison figures
Fix adaptive-weight metric calculation
```

避免：

```text
update
final
test
new code
```

上传流程：

```bash
git pull --rebase origin main
git status
git diff --check
git add <本次实验的明确文件>
git commit -m "Add ..."
git push origin main
```

不要使用 `git add .` 盲目加入所有文件。应明确列出本次要上传的目录和文件。

## 10. 新实验 README 模板

以后可以复制下面的结构：

````markdown
# YYYY-MM-DD：实验名称

## 学习位置

前一个方法是什么？本实验为什么接在它后面？

## 原方法哪里不够好

描述具体 limitation，不只写方法定义。

## 新方法如何改进

说明作者改变了采样、loss、网络、区域分解还是求解形式。

## 问题与实验设置

- PDE：
- 区域：
- IC/BC：
- 网络：
- 训练点：
- 优化器：
- 学习率：
- 迭代数：
- 随机种子：

## 运行

```bash
python train_or_run_script.py
```

## 核心结果

| 方法 | Relative L2 | PDE RMSE | IC RMSE | BC RMSE |
|---|---:|---:|---:|---:|
| Original | | | | |
| Improved | | | | |

![Direct comparison](results/main_comparison.png)

## 结果分析

解释哪里改善、哪里没有改善，以及原因。

## 与论文的差异

标明官方代码复现、论文配置移植或方法迁移实验。

## 下一步

列出一到三个有明确目的的后续实验。
````

## 11. 本仓库的学习主线

根目录 README 只维护阶段索引。每完成一个阶段，增加一条链接，并保持以下叙事顺序：

```text
基础方法
-> 观察失败现象
-> 分析 limitation
-> 学习针对性改进
-> 与原方法直接对比
-> 总结适用范围
-> 进入下一类方法
```

所有后续上传默认遵循本规范。如确实需要例外，应在该实验 README 中说明原因。

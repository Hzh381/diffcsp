# DiffCSP: Crystal Structure Prediction by Joint Equivariant Diffusion

DiffCSP 使用联合等变扩散模型同时生成晶体晶格和原子分数坐标。当前仓库在原始
CSPNet 去噪网络之外，新增了基于 EquiformerV3 的去噪网络。原 CSPNet 实现和
配置仍然保留，可以在任意训练命令中显式切换。

- 论文：[Crystal Structure Prediction by Joint Equivariant Diffusion](https://arxiv.org/abs/2309.04475)
- 官方代码：[jiaor17/DiffCSP](https://github.com/jiaor17/DiffCSP)
- EquiformerV3 参考实现：[atomicarchitects/equiformer_v3](https://github.com/atomicarchitects/equiformer_v3)

![Overview](fig/overview.png "Overview")

![Demo](fig/demo.gif "Demo")

## 1. 项目能力

| 任务 | 配置 | 说明 |
| --- | --- | --- |
| CSP 结构预测 | `model=diffusion_equiformer` | 默认配置，EquiformerV3 去噪网络 |
| CSP 结构预测 | `model=diffusion` | 原始 CSPNet 去噪网络 |
| 无条件晶体生成 | `model=diffusion_w_type` | 同时生成元素类型、坐标和晶格 |
| 性质引导优化 | `model=energy` | 训练时间相关的能量模型，用于优化采样 |
| 性质评估 | 评估脚本自动加载 `diffcsp/prop_models` | 评估稳定性、生成质量和覆盖度 |

支持的数据集：

| 数据集 | Hydra 名称 | 主要性质 | 条件生成 |
| --- | --- | --- | --- |
| Perov-5 | `perov_5` | 钙钛矿形成能 | 支持 |
| Carbon-24 | `carbon_24` | 每原子能量 | 支持 |
| MP-20 | `mp_20` | 每原子形成能 | 支持 |
| MPTS-52 | `mpts_52` | 每原子形成能 | 不支持无条件生成脚本 |

## 2. 目录结构

```text
DiffCSP/
├─ conf/
│  ├─ default.yaml                     # Hydra 总配置
│  ├─ data/                            # 数据集配置
│  ├─ model/                           # 模型配置
│  │  ├─ diffusion.yaml                # 原始 CSPNet
│  │  ├─ diffusion_equiformer.yaml     # EquiformerV3 CSP
│  │  ├─ diffusion_w_type.yaml         # 无条件生成
│  │  ├─ energy.yaml                   # 性质引导能量模型
│  │  └─ decoder/
│  │     ├─ cspnet.yaml
│  │     └─ equiformer_v3.yaml
│  ├─ optim/                           # 优化器和学习率调度器
│  └─ train/                           # Lightning Trainer 配置
├─ diffcsp/
│  ├─ run.py                           # 训练、验证和测试入口
│  ├─ common/
│  │  ├─ data_utils.py                 # 晶体图构建、PBC 和预处理
│  │  └─ constants.py                  # 性质归一化常数
│  ├─ pl_data/
│  │  ├─ dataset.py                    # CrystDataset / TensorCrystDataset
│  │  └─ datamodule.py                 # LightningDataModule
│  ├─ pl_modules/
│  │  ├─ cspnet.py                     # 原始 CSPNet 去噪网络
│  │  ├─ diffusion.py                  # CSP 扩散模型封装
│  │  ├─ diffusion_w_type.py           # 联合类型扩散生成模型
│  │  ├─ equiformer_v3/                # EquiformerV3 等变网络组件
│  │  ├─ energy_model.py               # 性质引导能量模型
│  │  ├─ gnn.py                        # DimeNet++ 性质预测骨干
│  │  └─ model.py                      # 监督性质预测模型
│  └─ prop_models/                     # 预训练性质预测权重
├─ scripts/
│  ├─ evaluate.py                      # CSP 结构预测/重构采样
│  ├─ generation.py                    # 无条件晶体生成采样
│  ├─ sample.py                        # 按化学式生成 CIF
│  ├─ optimization.py                  # 性质引导优化采样
│  ├─ compute_metrics.py               # CSP、生成和优化指标
│  └─ eval_utils.py                    # 模型、数据和评估公共函数
├─ data/                                # train.csv / val.csv / test.csv
├─ outputs/                             # Hydra 输出目录
├─ requirements.txt                     # Python 依赖
├─ .env.template                        # 环境变量模板
└─ README.md
```

## 3. 环境配置

### 3.1 基础要求

推荐使用项目验证过的环境：

- Python 3.8
- PyTorch 1.9.0
- PyTorch Geometric 1.7.2
- PyTorch Lightning 1.3.8
- CUDA 版本必须与 PyTorch 和 PyG 扩展 wheel 匹配

创建 Conda 环境：

```bash
conda create -n diffcsp python=3.8 -y
conda activate diffcsp
cd /path/to/DiffCSP
```

项目依赖版本记录在 [requirements.txt](requirements.txt)。
该文件只列出项目代码直接导入、训练/生成/评估实际需要的包；未使用的调试、
日志和仅作为传递依赖安装的包不会单独列出。

`pyxtal` 固定为 `0.5.9`，以保持与 `matminer==0.9.0` 所需的
`pandas~=1.5` 兼容。PyPI 上的 `pyxtal>=0.6` 要求 `pandas>=2`，两者不能同时
满足。

### 3.2 安装 PyTorch 和 PyG 扩展

`torch-scatter`、`torch-sparse` 是包含编译扩展的包，应先安装与 Python、PyTorch
和 CUDA 匹配的预编译 wheel。下面以 Python 3.8、PyTorch 1.9.0 为例。

CPU 环境：

```bash
pip install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 \
  -f https://data.pyg.org/whl/torch-1.9.0+cpu.html
```

CUDA 11.1 环境：

```bash
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 \
  -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
```

然后安装其余依赖：

```bash
pip install -r requirements.txt
```

如果已经通过 Conda 或系统环境安装了匹配版本的 PyTorch，不要重复安装错误的 CPU
或 CUDA wheel。EquiformerV3 额外依赖的 `e3nn==0.5.1` 已包含在
`requirements.txt` 中。

检查 GPU、PyTorch 和扩展是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import torch_scatter, torch_sparse, e3nn; print('extensions ok')"
```

### 3.3 配置环境变量

将模板复制为 `.env`：

```powershell
Copy-Item .env.template .env
```

Linux/macOS：

```bash
cp .env.template .env
```

在 `.env` 中填写绝对路径：

```text
PROJECT_ROOT=/path/to/DiffCSP
HYDRA_JOBS=/path/to/DiffCSP/outputs/hydra
WABDB_DIR=/path/to/DiffCSP/outputs/wandb
NUMBA_DISABLE_JIT=1
```

Windows 示例：

```text
PROJECT_ROOT=D:\code\workspace_pythoncode\thesisprojector\DiffCSP
HYDRA_JOBS=D:\code\workspace_pythoncode\thesisprojector\DiffCSP\outputs\hydra
WABDB_DIR=D:\code\workspace_pythoncode\thesisprojector\DiffCSP\outputs\wandb
NUMBA_DISABLE_JIT=1
```

`PROJECT_ROOT` 必须指向仓库根目录。当前 `run.py` 使用本地 `CSVLogger`，不要求
登录 WandB；`WABDB_DIR` 仍用于兼容原有 Hydra 配置。

### 3.4 准备数据

每个数据集应放在：

```text
data/<dataset>/
├─ train.csv
├─ val.csv
└─ test.csv
```

首次运行时会自动预处理并生成：

```text
data/<dataset>/train_ori.pt
data/<dataset>/val_ori.pt
data/<dataset>/test_ori.pt
```

如果 `.pt` 缓存不存在，程序会读取 CSV 并构建晶体图。首次预处理可能比后续训练
启动慢很多。当前配置文件中的 `preprocess_workers` 为 `1`，Windows 上更稳定；
Linux 上可以适当调大：

```bash
python diffcsp/run.py data=mp_20 ... data.preprocess_workers=8
```

## 4. 训练

所有训练命令都通过 `python diffcsp/run.py` 进入 Hydra 配置系统。核心参数使用
`key=value` 覆盖，例如 `data=mp_20` 和 `train.pl_trainer.gpus=1`。

### 4.1 快速检查配置

不启动训练，只打印解析后的 Hydra 配置：

```bash
python diffcsp/run.py data=mp_20 --cfg job --resolve
```

查看当前默认模型是否为 EquiformerV3：

```bash
python diffcsp/run.py data=mp_20 --cfg job --resolve | grep equiformer
```

PowerShell 可使用：

```powershell
python diffcsp/run.py data=mp_20 --cfg job --resolve | Select-String equiformer
```

### 4.2 使用 EquiformerV3 训练 CSP

默认配置已经使用 EquiformerV3，因此可以省略 `model=diffusion_equiformer`：

```bash
python diffcsp/run.py \
  data=mp_20 \
  expname=mp20_equiformer \
  train.pl_trainer.gpus=1
```

显式指定模型：

```bash
python diffcsp/run.py \
  data=mp_20 \
  model=diffusion_equiformer \
  expname=mp20_equiformer \
  train.pl_trainer.gpus=1
```

EquiformerV3 的显存占用明显高于原始 CSPNet。单卡 RTX 4090 建议先使用较小的
训练 batch：

```bash
python diffcsp/run.py \
  data=mp_20 \
  expname=mp20_equiformer_bs64 \
  train.pl_trainer.gpus=1 \
  data.datamodule.batch_size.train=64 \
  data.datamodule.batch_size.val=64
```

如果仍然 OOM，可以继续减小 batch，或减小
`model.decoder.attn_grid_resolution_list` 和
`model.decoder.ffn_grid_resolution_list`。

### 4.3 使用原始 CSPNet 训练

原始去噪网络没有被替换，显式选择 `model=diffusion` 即可：

```bash
python diffcsp/run.py \
  data=mp_20 \
  model=diffusion \
  expname=mp20_cspnet \
  train.pl_trainer.gpus=1
```

### 4.4 无条件晶体生成训练

```bash
python diffcsp/run.py \
  data=mp_20 \
  model=diffusion_w_type \
  expname=mp20_generation \
  train.pl_trainer.gpus=1
```

### 4.5 性质优化能量模型训练

```bash
python diffcsp/run.py \
  data=perov_5 \
  model=energy \
  expname=perov_energy \
  train.pl_trainer.gpus=1 \
  data.datamodule.batch_size.test=100
```

### 4.6 调试和常用覆盖参数

单步调试：

```bash
python diffcsp/run.py \
  data=mp_20 \
  expname=debug \
  train.pl_trainer.fast_dev_run=true
```

CPU 运行：

```bash
python diffcsp/run.py \
  data=perov_5 \
  expname=cpu_debug \
  train.pl_trainer.gpus=0 \
  train.pl_trainer.max_epochs=1
```

调整学习率、验证频率和数据加载进程：

```bash
python diffcsp/run.py \
  data=mp_20 \
  expname=mp20_tuned \
  train.pl_trainer.gpus=1 \
  optim.optimizer.lr=5e-4 \
  logging.val_check_interval=10 \
  data.datamodule.num_workers.train=4
```

Windows 上建议先把 DataLoader worker 设为 `0` 或 `1`：

```bash
python diffcsp/run.py \
  data=mp_20 \
  expname=mp20_windows \
  train.pl_trainer.gpus=1 \
  data.datamodule.num_workers.train=1 \
  data.datamodule.num_workers.val=1 \
  data.datamodule.num_workers.test=1
```

### 4.7 训练输出和续训

Hydra 输出目录默认为：

```text
${HYDRA_JOBS}/singlerun/<日期>/<expname>/
```

其中通常包含：

```text
*.ckpt               # 模型 checkpoint
hparams.yaml         # 本次运行完整配置
lattice_scaler.pt    # 晶格归一化参数
prop_scaler.pt       # 性质归一化参数
csv_logs/            # CSV 训练日志
```

`run.py` 会查找当前 Hydra 运行目录中的 `.ckpt` 并续训。需要复用相同
`expname` 和相同 Hydra 输出目录，例如：

```bash
python diffcsp/run.py \
  data=mp_20 \
  expname=mp20_equiformer \
  train.pl_trainer.gpus=1 \
  hydra.run.dir=outputs/hydra/singlerun/2026-11-13/mp20_equiformer
```

## 5. 测试与评估

测试脚本通常接收一个 **Hydra 运行目录** 作为 `--model_path`。该目录必须包含
`hparams.yaml`、`.ckpt` 以及 `lattice_scaler.pt` / `prop_scaler.pt`。

### 5.1 CSP 结构预测/重构评估

单次采样：

```bash
python scripts/evaluate.py \
  --model_path <run_dir> \
  --dataset mp_20

python scripts/compute_metrics.py \
  --root_path <run_dir> \
  --tasks csp \
  --gt_file data/mp_20/test.csv
```

多次采样：

```bash
python scripts/evaluate.py \
  --model_path <run_dir> \
  --dataset mp_20 \
  --num_evals 20

python scripts/compute_metrics.py \
  --root_path <run_dir> \
  --tasks csp \
  --gt_file data/mp_20/test.csv \
  --multi_eval
```

`scripts/evaluate.py` 默认使用推荐步长 `step_lr=-1`。也可以手动指定：

```bash
python scripts/evaluate.py \
  --model_path <run_dir> \
  --dataset mp_20 \
  --step_lr 1e-5
```

### 5.2 无条件生成评估

当前生成脚本支持 `perov_5`、`carbon_24` 和 `mp_20`：

```bash
python scripts/generation.py \
  --model_path <run_dir> \
  --dataset mp_20 \
  --num_batches_to_samples 20 \
  --batch_size 500

python scripts/compute_metrics.py \
  --root_path <run_dir> \
  --tasks gen \
  --gt_file data/mp_20/test.csv
```

### 5.3 按化学式生成 CIF

```bash
python scripts/sample.py \
  --model_path <run_dir> \
  --save_path outputs/samples \
  --formula BaTiO3 \
  --num_evals 20 \
  --batch_size 20 \
  --step_lr 1e-5
```

输出目录：

```text
outputs/samples/BaTiO3/BaTiO3_1.cif
outputs/samples/BaTiO3/BaTiO3_2.cif
...
```

### 5.4 性质优化

先训练能量模型，再进行优化采样：

```bash
python scripts/optimization.py \
  --model_path <energy_run_dir> \
  --uncond_path <diffusion_run_dir> \
  --test_samples 100 \
  --num_candidates 10 \
  --aug 50 \
  --step_lr 1e-5
```

计算优化指标：

```bash
python scripts/compute_metrics.py \
  --root_path <energy_run_dir> \
  --tasks opt
```

### 5.5 评估输出和 label

评估脚本会在 `--model_path` 下写出：

| 脚本 | 默认输出 |
| --- | --- |
| `evaluate.py` | `eval_diff.pt` |
| `generation.py` | `eval_gen.pt` |
| `optimization.py` | `eval_opt.pt` |
| `compute_metrics.py` | `eval_metrics.json` |

如果使用 `--label <name>`，输出会带后缀，例如：

```text
eval_diff_<name>.pt
eval_gen_<name>.pt
eval_opt_<name>.pt
eval_metrics_<name>.json
```

评估采样和指标计算要使用相同 `--label`。

## 6. 脚本说明

| 文件 | 用途 |
| --- | --- |
| `scripts/evaluate.py` | 在测试集上执行 CSP 结构预测/重构采样 |
| `scripts/generation.py` | 按数据集的原子数分布生成晶体 |
| `scripts/sample.py` | 给定化学式生成多个候选晶体并写出 CIF |
| `scripts/optimization.py` | 用能量模型引导无条件扩散模型进行性质优化 |
| `scripts/compute_metrics.py` | 计算匹配率、RMSD、有效性、覆盖率等指标 |
| `scripts/eval_utils.py` | 加载 checkpoint、数据集、性质预测器和公共评估函数 |

## 7. 重要命令参数

### 7.1 顶层 Hydra 参数

| 参数 | 可选值/示例 | 含义 |
| --- | --- | --- |
| `data=` | `perov_5`、`carbon_24`、`mp_20`、`mpts_52` | 数据集配置 |
| `model=` | `diffusion_equiformer`、`diffusion`、`diffusion_w_type`、`energy` | 模型配置 |
| `expname=` | `mp20_test` | 实验名，同时决定 Hydra 输出目录 |
| `optim.optimizer.lr=` | `1e-3`、`5e-4` | Adam 学习率 |
| `logging.val_check_interval=` | `5` | 每多少个 epoch 验证一次 |
| `train.deterministic=` | `true` / `false` | 是否固定随机种子 |
| `train.pl_trainer.gpus=` | `0`、`1` | CPU 或单卡 GPU |
| `train.pl_trainer.max_epochs=` | `100` | 覆盖数据配置中的最大 epoch 数 |
| `train.pl_trainer.fast_dev_run=` | `true` / `false` | Lightning 快速调试模式 |
| `train.pl_trainer.precision=` | `32` | 训练精度，默认使用 FP32 |
| `train.pl_trainer.accumulate_grad_batches=` | `1`、`4` | 梯度累积 |
| `train.pl_trainer.gradient_clip_val=` | `0.5` | 梯度裁剪阈值 |

### 7.2 数据参数

| 参数 | 含义 |
| --- | --- |
| `data.datamodule.batch_size.train` | 训练 batch size |
| `data.datamodule.batch_size.val` | 验证 batch size |
| `data.datamodule.batch_size.test` | 测试 batch size |
| `data.datamodule.num_workers.train` | 训练 DataLoader worker 数 |
| `data.datamodule.num_workers.val` | 验证 DataLoader worker 数 |
| `data.datamodule.num_workers.test` | 测试 DataLoader worker 数 |
| `data.preprocess_workers` | CSV 首次预处理进程数 |
| `data.prop` | 监督性质字段名 |
| `data.max_atoms` | 数据集最大原子数 |
| `data.train_max_epochs` | 数据配置默认的最大 epoch 数 |

### 7.3 扩散模型参数

| 参数 | 含义 |
| --- | --- |
| `model.time_dim` | 扩散时间步嵌入维度 |
| `model.latent_dim` | 额外 latent 维度，默认 `0` |
| `model.timesteps` | 扩散步数，默认 `1000` |
| `model.cost_coord` | 坐标损失权重 |
| `model.cost_lattice` | 晶格损失权重 |
| `model.cost_type` | 类型损失权重，仅 `diffusion_w_type` 使用 |
| `model.radius` | 周期近邻图搜索半径 |
| `model.max_neighbors` | 每个原子的最大邻居数 |

### 7.4 EquiformerV3 decoder 参数

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `model.decoder.hidden_dim` | `96` | 等变节点特征通道数 |
| `model.decoder.num_layers` | `4` | Transformer block 层数 |
| `model.decoder.lmax` | `2` | 最大球谐阶数 |
| `model.decoder.mmax` | `2` | 最大 SO(2) 阶数 |
| `model.decoder.attn_hidden_channels` | `48` | 注意力隐层通道数 |
| `model.decoder.ffn_hidden_channels` | `192` | FFN 隐层通道数 |
| `model.decoder.edge_channels` | `64` | 边特征通道数 |
| `model.decoder.num_radial_basis` | `32` | 高斯径向基数量 |
| `model.decoder.attn_grid_resolution_list` | `[6, 8]` | 注意力 S2 网格分辨率 |
| `model.decoder.ffn_grid_resolution_list` | `[8, 8]` | FFN S2 网格分辨率 |
| `model.decoder.norm_type` | `merge_layer_norm` | EquiformerV3 等变归一化方式 |
| `model.decoder.use_envelope` | `true` | 是否启用平滑截断包络 |
| `model.decoder.use_grid_mlp` | `true` | 是否使用 S2 网格 MLP |

Decoder 参数还可以在命令行直接覆盖：

```bash
python diffcsp/run.py \
  data=carbon_24 \
  expname=carbon_eqv3_small \
  train.pl_trainer.gpus=1 \
  model.decoder.num_layers=2 \
  model.decoder.hidden_dim=64 \
  model.decoder.attn_grid_resolution_list=[6,8] \
  model.decoder.ffn_grid_resolution_list=[8,8]
```

### 7.5 评估脚本参数

| 参数 | 脚本 | 含义 |
| --- | --- | --- |
| `--model_path` | 所有评估脚本 | 包含 checkpoint 和 `hparams.yaml` 的 Hydra 运行目录 |
| `--dataset` | `evaluate.py`、`generation.py` | 数据集名称 |
| `--root_path` | `compute_metrics.py` | 评估结果所在运行目录 |
| `--tasks` | `compute_metrics.py` | `csp`、`gen` 或 `opt` |
| `--gt_file` | `compute_metrics.py` | 测试集 CSV 路径 |
| `--num_evals` | `evaluate.py`、`sample.py` | 每个结构或组分的采样次数 |
| `--multi_eval` | `compute_metrics.py` | 多次采样时启用 |
| `--step_lr` | 采样脚本 | 扩散采样步长；`-1` 使用推荐值 |
| `--batch_size` | 生成脚本 | 采样 batch size |
| `--num_batches_to_samples` | `generation.py` | 生成多少个 batch |
| `--test_samples` | `optimization.py` | 参与优化的测试结构数 |
| `--num_candidates` | `optimization.py` | 每个结构保留的优化候选数 |
| `--aug` | `optimization.py` | 优化时的时间步增强参数 |
| `--label` | 评估脚本 | 输出文件后缀，便于多组实验对比 |

## 8. 常见问题

### 8.1 `ModuleNotFoundError`

重新安装依赖：

```bash
pip install -r requirements.txt
```

如果只缺少 EquiformerV3 依赖：

```bash
pip install e3nn==0.5.1
```

### 8.2 PyTorch 显示 `CUDA unavailable`

检查驱动和 wheel：

```bash
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

如果当前装的是 CPU 版，需要卸载后安装与显卡驱动匹配的 CUDA 版 PyTorch。

### 8.3 `torch-scatter` 或 `torch-sparse` 安装失败

不要直接使用不匹配的源码编译，优先安装对应 PyTorch/CUDA 版本的预编译 wheel：

```bash
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 \
  -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
```

将 `cu111` 替换为实际 CUDA 版本对应的索引。

### 8.4 `PROJECT_ROOT` 未配置

确认仓库根目录存在 `.env`，并包含：

```text
PROJECT_ROOT=<仓库绝对路径>
```

### 8.5 Windows 上 DataLoader 卡住

将：

```text
data.datamodule.num_workers.train
data.datamodule.num_workers.val
data.datamodule.num_workers.test
```

设为 `0` 或 `1`，不要直接使用 Linux 上的高 worker 数。

### 8.6 EquiformerV3 显存不足

依次尝试：

1. 降低 `data.datamodule.batch_size.train`。
2. 降低 `model.decoder.num_layers`。
3. 降低 `model.decoder.hidden_dim` 和 `ffn_hidden_channels`。
4. 降低 `attn_grid_resolution_list` / `ffn_grid_resolution_list`。
5. 使用 `train.pl_trainer.accumulate_grad_batches=4` 保持较小 batch。

### 8.7 首次训练很慢

首次运行会读取 CSV 并构建晶体图，生成 `*_ori.pt` 缓存。后续运行会直接复用缓存。
这是正常现象，不等同于每个 epoch 的训练时间。

## 9. 复现实验建议

建议每组实验使用不同的 `expname`，并将关键配置写入命令或单独的 Hydra override
文件，例如：

```bash
python diffcsp/run.py \
  data=mp_20 \
  expname=mp20_eqv3_l2_b64 \
  train.pl_trainer.gpus=1 \
  data.datamodule.batch_size.train=64 \
  optim.optimizer.lr=5e-4
```

训练完成后，将 `HYDRA_JOBS/singlerun/<日期>/<expname>` 作为评估脚本的
`--model_path`。评估阶段不要修改该目录中的 `hparams.yaml`、checkpoint 或 scaler。

## 10. 引用和致谢

如果本项目对你的研究有帮助，请引用：

```bibtex
@article{jiao2023crystal,
  title={Crystal Structure Prediction by Joint Equivariant Diffusion},
  author={Jiao, Rui and Huang, Wenbing and Lin, Peijia and Han, Jiaqi and Chen, Pin and Lu, Yutong and Liu, Yang},
  journal={arXiv preprint arXiv:2309.04475},
  year={2023}
}
```

本项目框架基于 [CDVAE](https://github.com/txie-93/cdvae)。Perov-5、Carbon-24
和 MP-20 数据来自 CDVAE，MPTS-52 数据来自其原始代码库。EquiformerV3 组件参考
[atomicarchitects/equiformer_v3](https://github.com/atomicarchitects/equiformer_v3)。

## 11. License

本项目使用 MIT License，详见 [LICENSE](LICENSE)。

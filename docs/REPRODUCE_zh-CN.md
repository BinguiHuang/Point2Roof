# Point2Roof 中文复现教程

本文针对 [Li-Li-Whu/Point2Roof](https://github.com/Li-Li-Whu/Point2Roof) 的代码版本 `ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad` 编写。检查日期：2026-11-10。

**验证范围：已拉取源码，检查训练、测试、数据加载和 CUDA 扩展，并通过 17 个 Python 文件的语法解析；当前编写环境未检测到 nvidia-smi，未编译扩展、下载完整数据或运行 GPU 训练/测试。下述环境是根据源码选择的起步配置，不是作者公布或本机实测通过的完整环境。**

## 1. 复现目标与路线

Point2Roof 从单栋建筑的机载 LiDAR 点云预测屋顶顶点和连接边。先跑官方权重测试，再做短训练检查，最后进行完整训练。

| 阶段 | 输入 | 可检查结果 |
| --- | --- | --- |
| 浏览已有结果 | 仓库 results/ | 16 栋建筑的点云、真值及方法对比 |
| 官方权重测试 | points.xyz、polygon.obj、checkpoint | 顶点/边 precision、recall 和顶点坐标偏差 |
| 从头训练 | 官方训练集 | 日志及每轮 checkpoint |
| 自有数据应用 | 分栋点云和独立推理适配 | 预测顶点及边；原始测试器仍要求真值 |

测试脚本不会自动导出预测 OBJ，仓库里的 results/ 是预先提供的成果。不要把这些文件当成你这次推理生成的结果。

## 2. 拉取和固定代码

你的 GitHub 已有 [BinguiHuang/Point2Roof](https://github.com/BinguiHuang/Point2Roof)。在准备运行的 Linux GPU 机器上：

```bash
git clone https://github.com/BinguiHuang/Point2Roof.git
cd Point2Roof
git remote add upstream https://github.com/Li-Li-Whu/Point2Roof.git
git fetch upstream
git rev-parse HEAD
git show ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad --stat
```

本教程以固定上游提交为依据。若要严格核对原始代码，可建立独立工作目录，避免切换当前文档分支：

```bash
git worktree add ../Point2Roof-upstream ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad
```

以下命令均在你选定的 Point2Roof 项目根目录执行。先检查仓库内有无已有目录或实验结果，实验标签尽量使用新名字。

## 3. 环境准备

源码包含 `THC/THC.h`、`THCState` 和旧式 Tensor API，直接使用最新 PyTorch 可能编译失败。建议先尝试 Linux、Python 3.8、PyTorch 1.8.1 + CUDA 11.1；该组合是保守的复现起点。PyTorch 安装命令依据[官方历史版本页面](https://docs.pytorch.org/get-started/previous-versions/)。

硬件需要支持该软件栈的 NVIDIA GPU。较新的 GPU 可能不适合 CUDA 11.1；应选兼容的旧 GPU/环境，或单独开展 CUDA 扩展迁移并记录差异。仓库未提供可据以保证的显存下限或运行耗时。

```bash
conda create -n point2roof python=3.8 -y
conda activate point2roof

python -m pip install "pip<24.1" "setuptools<60" wheel
python -m pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install numpy==1.23.5 scipy==1.10.1 scikit-learn==1.2.2 PyYAML==6.0.1 easydict==1.10 tqdm==4.66.5 ninja==1.11.1.1
```

这些辅助依赖版本用于约束旧环境，未经过本机端到端验证。`scikit-learn` 是安装名，导入名才是 `sklearn`。NumPy 固定到 1.23.5 是为了保留代码里的 `np.int`、`np.float`；使用 NumPy 1.24 及以上时需先修正这两个别名。

还需要 **完整 CUDA Toolkit（含 nvcc）和兼容的 C++ 编译器**。PyTorch wheel 带的 CUDA 运行库不能替代编译工具。安装位置根据你的机器调整：

```bash
export CUDA_HOME=/usr/local/cuda-11.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

nvidia-smi
nvcc --version
g++ --version
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); assert torch.cuda.is_available()"
```

`nvidia-smi` 显示的 CUDA 字样不代表本机 nvcc 版本。确认 Toolkit 与 PyTorch 的 CUDA 构建版本匹配，编译器也在 Toolkit 支持范围内。

### 编译 pc_util

```bash
cd pc_util
MAX_JOBS=2 python setup.py install
cd ..

python -c "import pc_util; print(pc_util.__file__); assert hasattr(pc_util, 'dbscan_wrapper')"
python train.py --help
python test.py --help
```

导入成功只是扩展加载检查，仍需后续实际 GPU 测试验证算子。

## 4. 下载数据与准备目录

作者 README 给出的数据入口：

- [百度网盘](https://pan.baidu.com/s/1Esbpnp30fWHA1_7eXwcYtQ)，提取码 `nou3`。
- [Google Drive](https://drive.google.com/drive/folders/1d8y2UxC9TUirgk6L1VbP-KXg3YbDgWYy?usp=sharing)。

上述链接来自作者，本文未验证下载完成后的压缩包内容与可用性。数据未包含在 Git 克隆中。作者还介绍了来自 RoofN3D 的真实建筑数据；合成与真实数据应分开记录测试结果。

每个样本至少包含：

```text
DATA_ROOT/
├── simulateRoof/
│   └── 000000/
│       ├── points.xyz
│       └── polygon.obj
├── simulateRoofTest/
│   └── 000000/
│       ├── points.xyz
│       └── polygon.obj
├── train.txt
└── test.txt
```

这是与源码清单匹配的示意目录，实际解压后请检查目录名。每行清单指向**样本目录**，不是 points.xyz 文件。用绝对路径最稳妥；相对路径按启动进程的工作目录解析。

数据格式有严格限制：

- points.xyz 为逐行 XYZ 数值；原始解析器使用 `split(' ')`，多空格、制表符、额外列或空行可能失败。
- polygon.obj 需包含 `v x y z` 和纯整数索引的面记录，索引从 1 开始；原始解析器会把所有非 v 行都当成面，不能直接接受常见的注释、材质、纹理或 `v/vt/vn` 记录。
- 点云和真值必须在同一坐标系。加载器按抽样点云计算一个全坐标共享的标量范围并归一化，不能随意更换归一化方法后仍宣称同设置复现。
- 每栋输入会随机抽样/补点到 1024 点；测试也有随机抽样。

### 重定位官方清单，保留原始划分

源码的 train.txt 与 test.txt 分别有 16000、1600 条样本路径，前缀是作者本机的 `/home/lili/ModelReconstruction/`。加载器读取的是 `--data_path/train.txt` 和 `--data_path/test.txt`，仅修改 --data_path 不会自动修复清单内容。

先设置数据根目录，以下示例需替换成真实路径：

```bash
export DATA_ROOT=/absolute/path/to/GithubDeepRoof
python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["DATA_ROOT"]).expanduser().resolve()
assert root.is_dir(), root
prefix = "/home/lili/ModelReconstruction/"
for split in ("train", "test"):
    src = Path(split + ".txt")
    target = root / src.name
    assert src.resolve() != target.resolve(), "请将数据清单写到独立数据根目录"
    paths = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        assert line.startswith(prefix), line
        sample = root / line[len(prefix):]
        for name in ("points.xyz", "polygon.obj"):
            assert (sample / name).is_file(), sample / name
        paths.append(str(sample))
    if target.exists():
        raise FileExistsError(f"{target} 已存在，请先检查或备份")
    target.write_text("\n".join(paths) + "\n")
    print(split, len(paths), target)
PY
```

若解压目录名不同，应修改映射到实际目录。不要为了让检查通过而把测试样本加入训练集。

### 数据加载检查

```bash
python - <<'PY'
import os
from dataset.data_utils import build_dataloader
from utils.common_utils import cfg_from_yaml_file
cfg = cfg_from_yaml_file("model_cfg.yaml")
loader = build_dataloader(os.environ["DATA_ROOT"], 1, cfg.DATA, workers=0, training=False)
batch = next(iter(loader))
print("points:", batch["points"].shape)
print("vectors:", batch["vectors"].shape, "edges:", batch["edges"].shape)
print("frame:", batch["frame_id"])
assert batch["points"].shape == (1, 1024, 3)
PY
```

这一步只检查首个样本的实际解析。完整数据仍可能包含坏样本；长任务前建议逐个验证格式、有限数值、非零坐标范围和面索引范围。

## 5. 优先跑官方预训练权重

原始 test.py 查找的是 `output/<test_tag>/*checkpoint_epoch_*.pth`。它**不会**自动使用项目根目录的权重，也不会查找 ckpt 子目录；更关键的是，没找到权重时仍会继续运行随机初始化模型。因此运行前必须确认权重存在，运行后确认日志出现加载信息。

```bash
mkdir -p output/official90
cp -n checkpoint_epoch_90.pth output/official90/checkpoint_epoch_90.pth
test -s output/official90/checkpoint_epoch_90.pth

python test.py --data_path "$DATA_ROOT" --gpu 0 --batch_size 1 --test_tag official90
```

日志：`output/official90/test/log.txt`。确认包含 `Loading parameters from checkpoint` 和 `Done`。若某标签目录有多个权重，源码按文件修改时间选最新文件，并非按轮数选最大值；建议每个测试标签只放一个目标权重。

输出包括：

- pts_recall、pts_precision：顶点召回率和精确率。
- pts_bias：匹配顶点反归一化后 X/Y/Z 的平均绝对偏差，单位继承原数据坐标单位。
- edge_recall、edge_precision：原始实现计算的边指标。

不要将一次测试输出直接当作论文表格复现成功。还需核对数据划分、权重、预处理、随机性以及指标定义。

## 6. 从头训练与测试自己的权重

先做短训练检查数据和反向传播，batch_size 可从 4 起步；若显存不足再降低。小批量可能触发 BatchNorm 的单样本问题，不能保证 batch_size=1 可以训练：

```bash
python train.py --data_path "$DATA_ROOT" --gpu 0 --batch_size 4 --epochs 2 --lr 0.001 --extra_tag smoke
```

**两轮只能检查前期顶点路径**。源码在零起始轮次 e > 5 时才启用边分支，即第 7 轮开始训练边。要验证完整训练路径，至少做 7 轮：

```bash
python train.py --data_path "$DATA_ROOT" --gpu 0 --batch_size 4 --epochs 7 --lr 0.001 --extra_tag smoke_full
```

上述 4 是诊断用批量，改变了原默认训练设置。正式实验保留原默认 batch_size=64、epochs=90、lr=0.001，并先确认机器资源能够支持：

```bash
python train.py --data_path "$DATA_ROOT" --gpu 0 --batch_size 64 --epochs 90 --lr 0.001 --extra_tag reproduce90
```

训练用 Adam，weight_decay=0.001；StepLR 每 20 轮乘 0.5，并设置学习率下限 1e-6。训练日志在 `output/reproduce90/log.txt`，checkpoint 在 `output/reproduce90/ckpt/`，默认仅保留最近 5 个。

同标签目录已有 checkpoint 时，源码会自动恢复训练。若要从头做新实验，使用新标签。断点恢复里的 scheduler 状态处理不等同于完整保存恢复 scheduler；严格对比建议先完成一次不中断的训练，并记录恢复行为。

测试自己的第 90 轮权重：

```bash
mkdir -p output/reproduce90_eval
cp -n output/reproduce90/ckpt/checkpoint_epoch_90.pth output/reproduce90_eval/checkpoint_epoch_90.pth
test -s output/reproduce90_eval/checkpoint_epoch_90.pth
python test.py --data_path "$DATA_ROOT" --gpu 0 --batch_size 1 --test_tag reproduce90_eval
```

复制这一步是为了适配训练/测试脚本目录不一致的行为。使用小批量或不同轮数训练时，请在结果里明确说明，不能与原默认设置混写。

## 7. 可视化与自有点云

无需运行神经网络，即可在 CloudCompare 等支持点云/OBJ 的工具中打开：

```text
results/Roof 1/points.xyz
results/Roof 1/ground_truth.obj
results/Roof 1/Point2Roof.obj
```

这只能验证和观察作者提供的结果。

要导出自己推理的线框，可在独立推理脚本中处理：取 batch["refined_keypoint"]，按 batch["keypoint"] 第一列的样本编号拆分；使用 batch["minMaxPt"] 将顶点反归一化；按边分支生成的顶点组合顺序取 edge_score > 0.5 的边；写出 OBJ 的 v 和 l 记录，索引从 1 开始。必须检查顶点不足 2 个时的处理。该导出功能不在上游测试入口中，本文未提供或验证新增导出代码。

自有 LAS/LAZ 或全城点云需要先提取单栋屋顶、清理噪声并转换到匹配格式；原始 test.py 还读取 polygon.obj 计算指标。没有真值时应编写不依赖标签的数据加载与推理入口，不能填造 polygon.obj 后把结果当成有效评估。

## 8. 源码限制与常见故障

| 现象 | 排查/处理 |
| --- | --- |
| No module named pc_util，或模块缺少 dbscan_wrapper | 编译扩展；检查是否误导入项目内同名目录，确认打印的是已安装扩展路径 |
| CUDA_HOME/nvcc 找不到 | 安装完整 CUDA Toolkit 并设置实际安装路径 |
| THC/THC.h 缺失、旧 Tensor API 编译错误 | 优先回到保守旧环境；若迁移到新 PyTorch，需要逐项适配，不能只删头文件就视为完成 |
| np.int / np.float 报错 | 使用 NumPy 1.23.5，或将对应标量别名分别改为 int / float 并记录改动 |
| 指定的 output 目录不存在 | 测试前创建标签目录并放入目标权重 |
| 测试能跑但没有权重加载日志 | 可能正在测随机模型，立即核对目录及文件名 |
| points.xyz 或 polygon.obj 找不到 | 检查数据根目录中的清单及每一行绝对路径 |
| DataLoader worker 错误难定位 | 临时把 build_dataloader 的 workers=16 改成 0，先在主进程定位坏样本 |
| shared memory / worker 异常 | 诊断后适当降低 worker 数，检查容器共享内存 |
| CUDA out of memory | 先降低训练批量并记录实验差异；测试保持 batch_size=1 |
| torch.cat 空列表、BatchNorm 单元素或除零 | 可能是预测聚类数太少、有效边太少或统计分母为零；保留失败样本并检查，不要直接跳过后宣称完整测试通过 |

**评估代码需特别审阅**：test_util.py 对顶点距离做匈牙利匹配后，用归一化距离 < 0.1 计数；但边评估使用全部 l_ind 组合，未沿用顶点距离过滤，而且当预测顶点多于真值时，组合数与 edge_score 数量可能不一致。这是从源码发现的逻辑风险，未通过 GPU 实验验证。原样复现应保留并披露该行为；修正版指标应单独报告，避免混用。

测试抽样仍使用 NumPy 随机数，脚本未统一设置随机种子。严格比较时，应记录 Python/NumPy/PyTorch/CUDA 的种子、worker 播种方式、抽样策略和确定性设置；不要承诺每次运行逐位一致。

## 9. 留存复现实验记录

```bash
mkdir -p reproduction_records
git rev-parse HEAD > reproduction_records/code_commit.txt
git diff > reproduction_records/local_changes.patch
python -m pip freeze > reproduction_records/pip_freeze.txt
nvidia-smi > reproduction_records/gpu.txt
nvcc --version > reproduction_records/nvcc.txt
sha256sum "$DATA_ROOT/train.txt" "$DATA_ROOT/test.txt" checkpoint_epoch_90.pth > reproduction_records/checksums.txt
```

另外记录数据压缩包来源/校验值、GPU 型号、实际命令、批量、轮数、种子及是否修改指标。数据清单的绝对路径随机器变化；跨机器对比划分时还应比较相对样本 ID。

| 实验 | 权重/轮次 | 数据划分 | 批量 | 顶点 P/R | 边 P/R | XYZ 偏差 | 指标是否修改 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 官方权重 | checkpoint_epoch_90.pth | 填写 | 1 | 待运行 | 待运行 | 待运行 | 否 |
| 从头训练 | 填写 | 填写 | 填写 | 待运行 | 待运行 | 待运行 | 填写 |

GPU 运行完成、日志确认加载正确权重、样本数与划分一致、没有静默跳过失败样本后，才填写实测结果。不要把未经运行的数字写入表格。

## 10. 依据与引用

- [上游 README（固定提交）](https://github.com/Li-Li-Whu/Point2Roof/blob/ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad/README.md)：论文信息、数据入口、官方权重说明。
- [train.py](https://github.com/Li-Li-Whu/Point2Roof/blob/ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad/train.py)、[train_utils.py](https://github.com/Li-Li-Whu/Point2Roof/blob/ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad/train_utils.py)：默认参数、恢复和分阶段训练。
- [test.py](https://github.com/Li-Li-Whu/Point2Roof/blob/ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad/test.py)、[test_util.py](https://github.com/Li-Li-Whu/Point2Roof/blob/ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad/test_util.py)：权重路径及原始评估逻辑。
- [dataset](https://github.com/Li-Li-Whu/Point2Roof/tree/ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad/dataset)、[pc_util](https://github.com/Li-Li-Whu/Point2Roof/tree/ab3d273f1dce45e7fc2e587bd17a7b8ec2fd3aad/pc_util)：数据格式、采样归一化及 CUDA 扩展。
- [PyTorch 官方历史版本](https://docs.pytorch.org/get-started/previous-versions/)：旧版安装方式。

科研使用请引用 Point2Roof 论文；使用真实数据还应按照上游要求引用 RoofN3D。保留原始 MIT LICENSE 和作者版权声明。

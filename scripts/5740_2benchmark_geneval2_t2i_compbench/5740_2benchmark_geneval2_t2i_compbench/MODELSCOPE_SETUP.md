# ModelScope 使用指南 (中国用户)

如果你无法连接到 HuggingFace (由于网络原因)，可以使用 ModelScope（魔塔社区）作为替代来下载模型。

## 快速开始

### 1. 安装 ModelScope

```bash
pip install modelscope
```

### 2. 使用 ModelScope 运行 Ablation 脚本

```bash
# 方式1: 使用环境变量
export USE_MODELSCOPE=true
bash scripts/run_ablation.sh

# 方式2: 直接设置变量
USE_MODELSCOPE=true bash scripts/run_ablation.sh
```

### 3. 使用 ModelScope 运行训练脚本

```bash
# 方式1: 使用环境变量
export USE_MODELSCOPE=true
python scripts/train.py --config configs/default.yaml

# 方式2: 直接设置变量
USE_MODELSCOPE=true python scripts/train.py --config configs/default.yaml
```

### 4. 使用 ModelScope 运行评估脚本

```bash
export USE_MODELSCOPE=true
python scripts/evaluate_benchmarks.py \
    --model_path deepseek-ai/Janus-Pro-1B \
    --output_dir evaluation_results/
```

## 支持的模型

ModelScope 支持大多数 HuggingFace 上的模型，包括：

- `deepseek-ai/Janus-Pro-1B`
- `deepseek-ai/Janus-Pro-7B`
- `openai/clip-vit-large-patch14`
- `Salesforce/blip-vqa-base`
- `google/owlv2-base-patch16-ensemble`

## 高级配置

### 自定义缓存目录

默认情况下，脚本使用与 `modelscope` CLI 相同的缓存目录 (`~/.cache/modelscope/hub`)，这样可以自动检测已下载的模型。

```bash
# 使用默认缓存目录 (推荐)
export USE_MODELSCOPE=true
bash scripts/run_ablation.sh

# 或使用自定义缓存目录
export MODELSCOPE_CACHE_DIR=/path/to/your/cache
export USE_MODELSCOPE=true
bash scripts/run_ablation.sh
```

### 下载 CLIP 模型（用于奖励计算）

如果训练使用 CLIP 奖励（默认配置），还需要下载 CLIP 模型：

```bash
# 下载 CLIP ViT-L-14 (openai) - 默认使用的模型
modelscope download --model timm/vit_large_patch14_clip_224.openai

# 如果需要其他 CLIP 模型
modelscope download --model timm/vit_base_patch32_clip_224.openai
modelscope download --model timm/vit_base_patch16_clip_224.openai
```

这些模型会被缓存到 `~/.cache/huggingface/hub` 或 `~/.cache/modelscope/hub`。

### 修复 CLIP 缓存路径问题

如果使用 `modelscope download` 下载了 CLIP 模型，但训练时仍然报错找不到模型，可能是因为：
- ModelScope 把模型下载到了 `~/.cache/modelscope/hub/`
- 但 `open_clip` 库需要 HuggingFace 特定的缓存结构：`models--<org>--<model>/snapshots/<hash>/`

解决方法：

```bash
# 方法1: 自动修复（创建正确的 HuggingFace 缓存结构）
python setup_hf_cache.py

# 或者使用包装脚本
bash fix_clip_cache.sh
```

这个脚本会：
1. 找到 ModelScope 下载的模型
2. 创建 HuggingFace 格式的缓存目录结构
3. 创建符号链接（不复制文件，节省空间）

修复后，再次运行训练脚本即可。

### 完整流程示例

```bash
# 1. 安装 modelscope
pip install modelscope

# 2. 下载 CLIP 模型
modelscope download --model timm/vit_large_patch14_clip_224.openai

# 3. 修复缓存路径
python setup_hf_cache.py

# 4. 运行训练
export USE_MODELSCOPE=true
bash scripts/run_ablation.sh
```

### 在 Python 代码中使用

```python
import os
os.environ['USE_MODELSCOPE'] = 'true'

# 导入 modelscope_helper 会自动配置 transformers 使用 ModelScope
from src.utils import modelscope_helper
modelscope_helper.setup_modelscope()

# 现在正常使用 transformers
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/Janus-Pro-1B")
```

### 直接下载模型到本地

```python
from src.utils.modelscope_helper import download_model

# 下载模型到本地
local_path = download_model(
    model_id="deepseek-ai/Janus-Pro-1B",
    cache_dir="./my_models"
)
print(f"模型已下载到: {local_path}")

# 使用本地路径加载模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(local_path)
```

## 工作原理

当 `USE_MODELSCOPE=true` 时：

1. **主模型下载**: `run_ablation.sh` 会使用 `modelscope.snapshot_download()` 从 ModelScope 下载 Janus-Pro 模型
2. **辅助模型下载**: Python 脚本会设置 `HF_ENDPOINT=https://www.modelscope.cn/hf`，使 HuggingFace transformers 自动使用 ModelScope 镜像
3. **本地缓存**: 所有模型都会缓存到本地，下次运行时直接从本地加载

## 常见问题

### Q: 为什么还是无法下载模型？

A: 请检查：
1. 是否已安装 modelscope: `pip install modelscope`
2. 是否正确设置了 `USE_MODELSCOPE=true`
3. 网络连接是否正常

### Q: 如何指定特定的模型版本？

A: 使用 `revision` 参数：

```python
from src.utils.modelscope_helper import download_model
local_path = download_model(
    model_id="deepseek-ai/Janus-Pro-1B",
    revision="v1.0"  # 指定版本
)
```

### Q: ModelScope 和 HuggingFace 的模型 ID 一样吗？

A: 大多数模型使用相同的 ID，例如 `deepseek-ai/Janus-Pro-1B` 在两个平台上是一样的。

## 环境变量参考

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `USE_MODELSCOPE` | `false` | 是否使用 ModelScope |
| `MODELSCOPE_CACHE_DIR` | `~/.cache/modelscope/hub` | ModelScope 缓存目录，默认与 `modelscope` CLI 相同 |
| `HF_ENDPOINT` | (自动设置) | HuggingFace 端点，自动设置为 ModelScope 镜像 |

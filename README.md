# Chinese-Mimi

Chinese-Mimi 是对 Moshi 模型的声码器 Mimi 进行了中文语料上的适配。本仓库主要包含 Mimi 的训练/推理代码 (支持分布式训练)。
此外，我们在 1w+ 小时的 [WenetSpeech4TTS](https://modelscope.cn/datasets/dukguo/WenetSpeech4TTS/files) 数据集上训练了多个版本的 Chinese-Mimi 供大家直接使用。

## 1. Preparation
### 1.1 数据准备

**a) 在根目录下创建 `/data` 文件夹**

```bash
mkdir data; cd data

mkdir -p WenetSpeech4TTS/Premium
```

将 WenetSpeech4TTS 的 Premium 数据集切片全部存放在 `data/WenetSpeech4TTS/Premium` 路径。

**b) 在根目录下创建 `/processed_data` 文件夹**

```bash
mkdir processed_data; cd processed_data

mkdir -p WenetSpeech4TTS/Premium
```

创建 `processed_data/WenetSpeech4TTS/Premium` 用于存放 Chinese-HuBERT 抽取的语义表征。



### 1.2 模型准备

在根目录下创建 `/models` 文件夹

```bash
mkdir models
```

下载预训练好的 [中文 HuBERT](https://huggingface.co/TencentGameMate/chinese-hubert-large)  模型存放至 `/models` 文件内。




### 1.3 环境配置

**a) 安装 fairseq**

首先降级 pip 到旧版本
```bash
python -m pip install "pip<24.1"
```

然后**按照顺序**安装指定版本的 omegaconf 和 hydra-core
```bash
pip install "omegaconf>=2.0.5,<2.1"

pip install "hydra-core>=1.0.7,<1.1"
```

最后，从源码安装 fairseq
```bash
git clone https://github.com/facebookresearch/fairseq
cd fairseq
# 回退到一个稳定的版本
git checkout v0.12.2
# 安装
pip install --editable .
```

**b) 安装ffmpeg**

具体操作可以参考 [知识库](https://dcnw4svfrk30.feishu.cn/wiki/HlbwwuSaMiMdUskPVAYccpycnqe?from=from_copylink)

**c) 安装 python 依赖库**

```bash
# flash attention 2
pip install flash-attn==2.3.3 --no-build-isolation --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 其他依赖库
pip install -r requirements.txt
```



## 2. Mimi

### 2.1 准备音频元文件

在根目录执行以下命令:
```bash
python -m src.data.prepare_audiodata
```

### 2.2 利用预训练的 HuBERT 抽取语义特征

脚本在 `src/data/extract_semantic_rep.py` 

首先准备好原始数据和模型，并修改对应的路径。

在根目录执行以下命令：
```bash
python -m src.data.extract_semantic_rep --meta_path data/wenetspeech4tts_premium_train.jsonl
python -m src.data.extract_semantic_rep --meta_path data/wenetspeech4tts_premium_valid.jsonl
```

### 2.3 开启训练

训练配置在 `configs/mimi.yaml`

最好先执行 `src/utils/len_count.py` 来统计训练集中 duration 的分布，根据分布设置 mimi.yaml 中的 `segment_duration` 参数。 此外，`batch_size` 务必设置为偶数。

可以设置 `src/utils/compile.py` 中参数 `_compile_disabled=False` 来开启编译加速

执行 `sh train.sh` 采用 DDP 分布式训练

执行 `python -m src.main` 采用单卡训练

训练输出会保存到 `outputs` 路径 (`logs` 记录日志 `save` 保存模型)

### 2.4 测试效果

**测试验证集**

创建测试集以及临时目录保存重构音频
```bash
mkdir tmp
mkdir data/WenetSpeech4TTS/test
```

> 需要随机从验证集中随机选一些测试样例放到 `/test` 文件中。

测试脚本

```bash
python -m test.mimi --epoch 20 --model-id 20241211_202021
```

**gradio服务**

```bash
python -m src.gradio
```

支持麦克风录制音频，也可以自己上传音频文件。


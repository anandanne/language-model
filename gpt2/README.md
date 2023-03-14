## GPT2

`GPT2` 模型来自 `OpenAI` 的论文[《Language Models are Unsupervised Multitask Learners》](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)无监督的多任务学习语言模型。

+ 尽管目前很多有监督学习 `NLP` 模型效果已经很好，但都需要有针对单个任务训练使用大量有标注数据训练，当目标的分布稍有变化则不能继续使用，因此只能在狭窄的领域中起作用。

+ `GPT2` 希望通过海量数据和庞大的模型参数训练出一个类似百科全书的模型，无需标注数据也能解决具体问题。

+ `GPT2` 希望在完全不理解词的情况下建模，以便让模型可以处理任何编码的语言。

+ `GPT2` 主要针对 `zero-shot` 问题。它在解决多种无监督问题时有很大提升，但是对于有监督学习则差一些。

无监督学习和有监督学习的效果对比，就像两个小孩子学习，一个博览群书，但看的不一定考；另一个专看考点，定点优化。结果就是一个在考试里面成绩更好，另一个能力更强，能解决各种问题，尤其适用于无确定答案的问题。它们在不同的领域各具特长。

`GPT2` 的结构类似于 `GPT1` 模型，仍然使用单向的 `Transformer` 模型，只做了一些局部修改：如将归一化层移到 `Block` 的输入位置；在最后一个自注意力块之后加了一层归一化；增大词汇量等等。

与之前的实现方法最大的不同是：`GPT2` 的训练数据在数量、质量、广泛度上都有大幅度提高：抓取了大量不同类型的网页，并且经过筛选去重生成高质量的训练数据，同时训练出体量更巨大的模型。

在 `Pretrain` 部分基本与 `GPT` 方法相同，在第二阶段的 `Fine-tuning` 具体任务，当问题的输入和输出均为文字时，只需要用特定方法组织不同类型的有标注数据即可代入模型，如对于问答使用“问题+答案+文档”的组织形式，对于翻译使用“英文+法文”形式。用前文预测后文，而非使用标注数据调整模型参数。这样既使用了统一的结构做训练，又可适配不同类型的任务。虽然学习速度较慢，但也能达到相对不错的效果。

## 训练

```commandline
python run_gpt2.py \
    --model_name_or_path uer/gpt2-chinese-cluecorpussmall \
    --train_file data/train.txt \
    --validation_file data/valid.txt \
    --cache_dir data \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 10 \
    --save_total_limit 2 \
    --do_train \
    --do_eval \
    --save_steps 1000 \
    --logging_steps 1000 \
    --output_dir outputs/gpt2
```

训练参数含义：

+ `model_name_or_path`：模型文件路径，包含配置、权重、词表等

+ `train_file`：训练集文件名或文件名列表

+ `validation_file`：验证集文件名或文件名列表

+ `cache_dir`：数据和模型的缓存路径

+ `per_device_train_batch_size`：训练时单个 `gpu` 设备的批量大小

+ `per_device_eval_batch_size`：验证时单个 `gpu` 设备的批量大小

+ `num_train_epochs`：训练轮次

+ `save_total_limit`：最多保存的模型个数

+ `do_train`：是否进行训练

+ `do_eval`：是否进行验证评估

+ `save_steps`：每隔多少步保存一次模型

+ `logging_steps`：每隔多少步打印一次日志

+ `output_dir`：模型结果保存路径

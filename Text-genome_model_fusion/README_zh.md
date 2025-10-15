# Text-genome model fusion
## 1. 项目概述
本项目是一个基于DNA模型+文本模型的DNA序列分析系统，主要用于基因变异效应预测和疾病关联分析。项目结合了DNA序列编码器和大型语言模型，实现了对DNA序列的深度理解和分析。
## 2. 数据分析
### 2.1 输入数据格式
#### KEGG数据集
● 文件格式: JSON格式

● 数据结构:
  ```json
  {
    "question": "染色体信息和通路网络定义的问题描述",
    "answer": "疾病名称（如：cushing syndrome, parkinson's disease, amyotrophic lateral sclerosis）",
    "reasoning": "详细的推理步骤，包含10个步骤的生物学分析",
    "reference_sequence": "参考DNA序列（大写字母，去除空格）",
    "variant_sequence": "变异DNA序列（大写字母，去除空格）"
  }
  ```
  ### 2.2 输出数据格式
#### 2.2.1 模型输出
● 文本生成: 模型生成包含推理过程和最终答案的文本

● 格式:
  ```
  <|im_start|>assistant
  [推理内容]
  Answer: [最终答案]<|im_end|>
  ```

#### 2.2.2 评估结果
● CSV格式: 包含预测结果和真实标签的对比

● 列结构:

    ○ ground_truth: 真实标签
    ○ pred_label: 预测标签
    ○ generated_text: 完整生成文本

## 3.数据预处理流程
### 3.1 DNA序列预处理
#### 3.1.1 序列标准化
● 大小写转换: 所有DNA序列转换为大写字母

● 空格去除: 移除序列中的空白字符

● 序列截断: 使用```truncate_dna```函数从两端截断序列
 ```python
  def truncate_dna(example, truncate_dna_per_side=1024):
      # 从序列两端各截断1024个碱基对
      # 如果序列太短，返回中间部分
  ```

#### 3.1.2 序列标记化
● OURGEN标记器: 使用字符级标记器处理DNA序列

● 特殊标记: 添加<|dna_start|>, <|dna_pad|>, <|dna_end|>标记

● 序列长度限制: 最大长度2048个token
### 3.2 文本预处理
#### 3.2.1 对话格式转换
● 多模态格式: 结合DNA序列和文本的对话格式

● 角色定义:

    ○ user: 包含DNA序列和问题
    ○ assistant: 包含推理过程和答案
#### 3.2.2 模板应用
● 聊天模板: 使用自定义的聊天模板格式化输入

● 特殊标记处理: 正确处理```<|im_start|>```和```<|im_end|>```标记

### 3.3 数据加载和批处理
#### 3.3.1 数据集分割
● 训练集: 80%

● 验证集: 10%

● 测试集: 10%
#### 3.3.2 批处理函数
● qwen_dna_collate_fn: 专门为Qwen DNA模型设计的批处理函数

● 标签掩码: 只对助手回复部分计算损失

● 填充处理: 左填充策略，添加特殊标记

## 4. 测评指标方案
### 4.1 分类指标
#### 4.1.1 基础指标
● 准确率 (Accuracy): 正确预测的样本占总样本的比例

● 精确率 (Precision): 宏平均精确率，所有类别精确率的平均值

● 召回率 (Recall): 宏平均召回率，所有类别召回率的平均值

● F1分数 (F1-Score): 宏平均F1分数，精确率和召回率的调和平均
#### 4.1.2 计算方式
```python
# 使用sklearn的classification_report
report_dict = classification_report(
    y_true, y_pred, 
    labels=labels, 
    output_dict=True, 
    zero_division=1
)

# 提取宏平均指标
macro_metrics = report_dict['macro avg']
Accuracy = accuracy_score(ground_truth, pred_label)
Precision = macro_metrics['precision']
Recall = macro_metrics['recall'] 
F1_score = macro_metrics['f1-score']
```
### 4.2 评估流程
#### 4.2.1 验证阶段评估
● 实时生成: 在验证集上生成文本

● 答案提取: 从生成文本中提取答案部分

● 指标计算: 计算分类指标并记录到日志
#### 4.2.2 测试阶段评估
● 完整评估: 在测试集上进行完整评估

● 结果保存: 将预测结果保存为CSV文件

● 指标汇总: 生成详细的分类报告
### 4.3 特殊处理
#### 4.3.1 答案提取
● 正则匹配: 使用```extract_single_entry```函数提取答案

● 格式处理: 处理包含```<think>```前缀的预测结果

● 错误处理: 对无法提取的答案返回NaN
#### 4.3.2 多类别支持
● 动态标签: 根据真实标签动态确定类别

● 零除处理: 使用zero_division=1处理无预测的情况

● 宏平均: 使用宏平均确保所有类别平等对待

## 5. 模型架构
### 5.1 核心组件
● DNA编码器: Evo2模型用于DNA序列编码

● 文本模型: Qwen用于文本理解和生成

● 投影层: 将DNA特征投影到文本嵌入空间

● LoRA适配: 使用LoRA进行参数高效微调
### 5.2 训练策略
● 冻结策略: 可选择冻结DNA编码器或文本模型

● 混合精度: 使用DeepSpeed策略进行高效训练

● 梯度累积: 支持大批次训练

● 混合精度: 支持大批次训练


## 6. 使用
### 6.1 环境准备
该工程需要在装有GPU的机器上运行，请先检测机器上是否安装了CUDA
```python
# 检测环境
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
```
工程中提供了requirements.txt文件，用户可以通过以下命令安装依赖：
```shell
pip install -r requirements.txt
```
### 6.2 模型训练
本工程提供了两种运行方式，shell脚本和jupyter notebook。
#### 6.2.1 Shell脚本
用户可以通过运行`sh_train.sh`开始模型训练。
#### 6.2.2 Notebook
用户也可以通过运行notebook文件`user_case.ipynb`来开始模型训练。
### 6.3 模型部署
本工程提供了fastapi的方式来部署模型。用户可以通过运行脚本`start_simple.sh`来启动服务，注意要修改其中的模型参数。然后通过运行脚本`test_simple_api.py`来测试服务。
# 10 · 场景与手撕题

> 核心考点：softmax 手撕、Self-Attention 矩阵计算、Top-k 采样、BPE 核心逻辑、系统设计题

---

### Q1 【实战】手撕 softmax（带数值稳定）

- **题目**：用 Python 实现数值稳定的 softmax 函数。
- **答案要点**：
  1. 原始 softmax：`softmax(x_i) = exp(x_i) / Σ exp(x_j)`
  2. 数值不稳定问题：x 值很大时 exp(x) 会溢出（inf）；x 值很小时 exp(x) 下溢为 0。
  3. 稳定实现：先减去最大值 `x_max = max(x)`，然后 `exp(x - x_max)`——因为 `exp(x - c) / Σ exp(x_j - c) = exp(x) / Σ exp(x_j)`，数学上等价，但减去最大值后最大的 exp(0)=1，不会溢出。
  4. 代码：
     ```python
     import numpy as np
     def softmax(x):
         x = x - np.max(x, axis=-1, keepdims=True)
         exp_x = np.exp(x)
         return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
     ```
  5. 为什么减最大值不改变结果：softmax 对输入加常数是不变的（分子分母都乘同一个常数）。
- **追问点**：
  - 如果输入是二维矩阵怎么处理？（答：按行做，axis=-1 + keepdims 保持维度）
  - 推理时为什么不用 log_softmax？（答：log_softmax 更稳定，计算交叉熵时直接用；但 softmax 更直观）

---

### Q2 【实战】手撕 Self-Attention 矩阵计算

- **题目**：用 NumPy/PyTorch 实现 Scaled Dot-Product Attention。
- **答案要点**：
  1. 公式：`Attention(Q, K, V) = softmax(QK^T / √d_k) · V`
  2. 代码（PyTorch）：
     ```python
     import torch
     import torch.nn.functional as F

     def self_attention(Q, K, V, mask=None):
         d_k = Q.size(-1)
         scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
         if mask is not None:
             scores = scores.masked_fill(mask == 0, -1e9)
         attn_weights = F.softmax(scores, dim=-1)
         output = torch.matmul(attn_weights, V)
         return output, attn_weights
     ```
  3. 关键点：(a) 除以 √d_k；(b) mask 位置填 -inf 再 softmax；(c) 输出是 V 的加权和。
  4. Causal mask：下三角矩阵，保证每个位置只能看到左边。
- **追问点**：
  - 多头怎么实现？（答：把 d_model 拆成 n_heads 份，每份独立做 attention，最后 concat）
  - Q/K/V 是怎么来的？（答：输入 embedding 分别乘三个投影矩阵 W_q, W_k, W_v）

---

### Q3 【实战】手撕 Top-k 采样

- **题目**：实现 Top-k 采样——只从概率最高的 k 个 token 中采样。
- **答案要点**：
  1. 流程：(a) 拿到 logits；(b) 按概率排序，只保留 Top-k 个；(c) 其他概率设为 -inf；(d) 重新 softmax；(e) 采样。
  2. 代码：
     ```python
     import torch
     import torch.nn.functional as F

     def top_k_sampling(logits, k=50, temperature=1.0):
         logits = logits / temperature
         top_k = min(k, logits.size(-1))
         # 只保留 top_k，其余设为 -inf
         indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
         logits[indices_to_remove] = float('-inf')
         probs = F.softmax(logits, dim=-1)
         next_token = torch.multinomial(probs, num_samples=1)
         return next_token
     ```
  3. Temperature：控制随机性——温度越低越确定（greedy），越高越随机。
  4. Top-p（核采样）：比 Top-k 更自适应——累计概率达到 p 就停，动态调整候选数量。
- **追问点**：
  - Top-k 和 Top-p 哪个更好？（答：Top-p 更自适应，分布平缓时候选多，分布尖锐时候选少；实际生产常用 Top-p）
  - 为什么需要采样？直接取 argmax 不好吗？（答：argmax 永远取最高概率，输出会很死板、重复；采样增加多样性，但太随机又会胡言乱语；要在确定性和多样性间平衡）

---

### Q4 【实战】手撕 BPE 分词器核心逻辑

- **题目**：实现 BPE 训练和编码的核心逻辑。
- **答案要点**：
  1. 训练阶段：统计词频 → 初始化字符级 → 合并最高频对 → 重复直到词表大小。
  2. 编码阶段：按学到的合并规则，把输入词逐对合并。
  3. 核心代码：
     ```python
     from collections import Counter

     def train_bpe(texts, vocab_size=1000):
         # 初始化：每个词拆成字符 + 结尾符
         word_freq = Counter(texts)
         vocab = set(char for word in word_freq for char in word)
         merges = {}

         while len(vocab) < vocab_size:
             # 统计所有相邻对
             pairs = Counter()
             for word, freq in word_freq.items():
                 chars = tuple(word)
                 for i in range(len(chars)-1):
                     pairs[(chars[i], chars[i+1])] += freq
             if not pairs:
                 break
             # 合并最高频对
             best = pairs.most_common(1)[0][0]
             merges[best] = ''.join(best)
             vocab.add(''.join(best))
             # 更新词表（简化版）
         return vocab, merges

     def encode_bpe(word, merges):
         tokens = list(word)
         while True:
             # 找到可以合并的对
             pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
             # 按 merges 优先级合并（简化：找第一个能合并的）
             merged = False
             for pair, merge_token in merges.items():
                 if pair in pairs:
                     idx = pairs.index(pair)
                     tokens = tokens[:idx] + [merge_token] + tokens[idx+2:]
                     merged = True
                     break
             if not merged:
                 break
         return tokens
     ```
  4. 关键点：合并规则是训练时定死的；编码时按优先级依次合并。
- **追问点**：
  - 生产级 BPE 和这个简化版差在哪？（答：性能优化（预统计、高效查找）、byte-level 处理、特殊字符、并行化；逻辑是一样的）

---

### Q5 【实战】设计一个智能客服系统的技术方案

- **答案要点**：
  1. 需求分析：自动回答用户问题、转人工、查订单、处理退换货。
  2. 技术架构：
     - (a) 意图识别：判断用户要干什么（查订单/投诉/咨询）。
     - (b) RAG 知识库：产品文档、FAQ、政策。
     - (c) 工具调用：查订单 API、退换货 API。
     - (d) LLM：整合信息生成回答。
     - (e) 转人工：解决不了的问题转人工客服。
  3. 流程：用户输入 → 意图识别 → 检索知识库/调用工具 → LLM 生成回答 → 不满意转人工。
  4. 评估：解决率、转人工率、用户满意度。
- **追问点**：
  - 怎么保证回答不出错？（答：关键信息必须从工具/知识库来，不能让模型自由发挥；加事实校验；高风险操作必须人工确认）

---

### Q6 【实战】设计一个企业知识库问答系统

- **答案要点**：
  1. 文档接入：多格式解析（PDF/Word/PPT/网页）→ 清洗 → 分块 → 向量化 → 入库。
  2. 检索：用户 query → 混合检索（向量+BM25）→ Rerank → Top-K 文档。
  3. 生成：拼接 prompt（系统指令 + 检索到的文档 + 用户问题）→ LLM 生成回答 + 引用来源。
  4. 权限：多租户隔离、文档级 ACL。
  5. 评估：召回率、忠实度、用户反馈。
  6. 迭代：bad case 收集 → 优化 chunking/检索/prompt。
- **追问点**：
  - 文档更新了怎么办？（答：增量索引——新文档单独向量化入库；修改的文档重新处理；定期全量重建）

---

### Q7 【实战】设计一个代码审查 Agent

- **答案要点**：
  1. 输入：Git diff（变更的代码）、PR 描述。
  2. 流程：
     - (a) 理解变更：这个 PR 改了什么、为什么改。
     - (b) 检查：bug、性能问题、安全问题、风格规范。
     - (c) 输出：结构化的审查意见（文件/行号/问题/建议）。
  3. 工具：可以调用代码搜索、跑测试、查文档。
  4. 边界：Agent 给建议，人做最终决策；不要自动改代码。
  5. 评估：漏检率、误报率、开发者满意度。
- **追问点**：
  - 怎么让 Agent 理解整个项目上下文？（答：RAG 检索相关代码文件；把项目结构、架构文档塞进 prompt；必要时调用代码搜索工具）

---

### Q8 【实战】设计一个多语言翻译+润色的 pipeline

- **答案要点**：
  1. 需求：输入原文 → 翻译成目标语言 → 润色到专业风格。
  2. Pipeline：
     - (a) 语种检测：判断输入是什么语言。
     - (b) 初翻：大模型翻译（prompt 中指定风格要求）。
     - (c) 润色：再让模型检查语法、流畅度、专业术语一致性。
     - (d) 术语库：维护公司专属术语表，确保翻译一致。
  3. 质量控制：关键内容人工审核；建立反馈闭环。
- **追问点**：
  - 怎么保证术语一致性？（答：术语表作为 few-shot 示例塞进 prompt；或者微调模型专门学术语）

---

### Q9 【实战】怎么评估一个新模型能不能替代现有线上模型？

- **答案要点**：
  1. 离线评估：
     - (a) 标准 benchmark：MMLU、C-Eval、GSM8K、HumanEval 等。
     - (b) 业务评测集：用业务上真实的 100~500 条 case 跑对比，人工或 LLM-as-Judge 打分。
     - (c) 延迟/成本：首 token 延迟、吞吐量、单次调用成本。
  2. 线上灰度：
     - (a) 先切 1%~5% 流量到新模型，监控核心指标（用户满意度、投诉率、转人工率）。
     - (b) 逐步放量；发现问题随时回滚。
  3. 决策标准：效果不能下降太多（<2%）、成本/延迟要有优势、灰度数据稳定。
- **追问点**：
  - 为什么不能只看 benchmark？（答：benchmark 测的是通用能力，业务场景可能完全不同；一定要用业务数据做评测）

---

### Q10 【实战】面试题：为什么大模型会"一本正经地胡说八道"？怎么缓解？

- **答案要点**：
  1. 根本原因：大模型是概率生成模型，优化目标是"生成像人话的文本"，不是"生成正确的文本"；在不知道答案时，它仍然会根据语言模式编一个看起来合理的回答。
  2. 具体原因：
     - (a) 训练数据中没有这个知识——模型"不知道自己不知道"。
     - (b) 训练数据中有错误信息——模型学了错误的知识。
     - (c) 概率生成导致组合错误——每个词都合理，组合起来不对。
  3. 缓解方法（分层）：
     - (a) 数据层：更高质量的训练数据。
     - (b) 对齐层：RLHF 中奖励"诚实承认不知道"的行为。
     - (c) 外挂层：RAG 让模型基于真实文档回答。
     - (d) 推理层：CoT 多步推理、Self-Consistency 多次采样投票。
     - (e) 后处理：事实核查模型校验。
  4. 根本上：幻觉无法完全消除，只能降低；这是概率生成范式的固有问题。
- **追问点**：
  - 为什么模型不知道自己不知道？（答：因为模型没有"元认知"能力——它不知道自己知道什么、不知道什么；它只是根据概率生成最合理的下一个 token）

---

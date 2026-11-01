# papers

这里收集目前能够打开的论文，按方向整理，用来看业界在往哪走，以及哪些说法可以借来做模因和调优。

收录标准：2026-10-24 打开过全文或出版页。只保存题名、作者、年份和链接，不把 PDF 拷进仓库。付费墙只标成“摘要可见”。公式已经接到代码里的那一批仍在 `doc/v8.3/memetics_library/`，这里会指过去，不把同一条公式再写一遍。

| 方向 | 文件 |
| --- | --- |
| 机器学习 | [machine_learning.md](machine_learning.md) |
| 深度学习 | [deep_learning.md](deep_learning.md) |
| 计算语义 | [computational_semantics.md](computational_semantics.md) |
| 传播 | [transmission.md](transmission.md) |

机器可读清单是 [catalog.json](catalog.json)。

## 现在能借用的，先停在阅读

调优这一侧，随机搜索是基线，网格搜索在多数超参数并不起作用时浪费预算（Bergstra、Bengio 2012）。评估很贵时再用贝叶斯优化（Snoek、Larochelle、Adams 2012）。可以提前停止的试验用 Hyperband（Li 等 2018）。调参的成绩必须放在外层验证上，否则调得越久，验证分越好看，泛化不一定更好（Bischl 等的综述）。自适应优化里，权重衰减要和梯度更新分开（Loshchilov、Hutter 2019）。模型变大和数据变多要一起算，不能只加参数（Hoffmann 等 2022；Li、Kudugunta、Zettlemoyer 2025）。

模因这一侧，别人的分歧已经写清楚了。选择论把稳定归因于高保真复制加选择。吸引论把稳定归因于转换时的偏向，复制只是特例（Claidière、Scott-Phillips、Sperber 2014，见模因库）。Acerbi、Charbonneau、Miton、Scott-Phillips 2021 进一步说明：种群可以在没有复制、也没有选择的情况下保持稳定，靠的是收敛式转换。这和我们测到的情形同一类：草稿彼此略近，但没有一份跨过复制阈值。Edmonds 2005 说明把索引叫做模因并不使它变成复制子。Kempe、Kleinberg、Tardos 2015 的影响最大化是网络上的子模覆盖，和“一句文本是不是复制子”不是同一个问题，不拿来改 `judgeCopy`。

大模型全量微调不是这条收集要启动的操作。LoRA 和 DPO 都假定已经有一个看过主序列的预训练模型。现场 1072 参数的网络装不下源码 diff，这一点不因多读一篇缩放定律而改变。

## 2018-01-01 至 2026-10-24 的范围

条件是四条里的任意一条：CCF A/B/C，SCI，EI，IEEE 或 ACL 出版。原文写的是 “ti收录”。公开检索里没有名为 TI 的收录库，这里按 EI（Compendex）理解。SCI 的主列表属于科睿唯安，EI 的来源列表属于 Elsevier，两边都没有可整库下载的公开接口，所以不能把“SCI 收录的全部论文”或“EI 收录的全部论文”写成已经收齐。

能公开枚举、并且能对上刊物本身的，是 CCF 2022 目录。目录转录在 `ccf_2022.json`（645 个 A/B/C 条目，其中会议 364、期刊 281）。`harvest_open.py` 先把会刊名解析成 OpenAlex 的 source id：类型要和目录一致，去掉 the/of/conference/ieee 这类泛词之后，特异词的 Jaccard 不低于 0.75，并且不用 workshop、forum、demo、student 会去顶主会。解析通过之后，只拉这个 source 在日期窗里、且 `is_paratext:false` 的作品。六个已知会刊（TOCS、CVPR、ICML、ACL、NeurIPS、SIGIR）在开爬之前都对上了；这条门槛不按后面的漏检数再改。

IEEE 与 ACL 的出版方总数走 Crossref。IEEE 是 member 263，ACL 是 member 1643。2026-10-24 查到的日期窗总数是 IEEE 2,945,079、ACL 56,548，写在 `collected/scope_counts.json`。ACL 的作品会逐页写入 `collected/acl_works.jsonl`。IEEE 这份清单不下载。CCF 目录里的 IEEE 会和刊会随 source 抓取进入 `collected/ccf_works.jsonl`。没有 SCI 或 EI 标记，因为接口不提供这个字段。

`collected/topics.jsonl` 是更早一次 Crossref 词袋检索留下的 854 行。那次检索报告的总数很大（“cultural attraction theory” 为 1,098,349），因为词袋会命中任何一个共享词的 IEEE 记录。按题名重新核对后，854 行里只有 3 行盖住了查询词的六成以上。那 3 行在 `collected/topic_title_match.txt`。不要把 `topics.jsonl` 或 `topics_unfiltered_partial.jsonl` 当成这份目录。

OpenAlex 匿名额度在写完前 12 个会刊（14,648 条，saved 与 reported 相等）后返回 429，到 ACL 清单写完时仍然是 429。剩下的改走 Crossref。期刊用 ISSN，saved 与 reported 对齐。会议先试简称和去掉 IEEE/ACM 前缀后的会名，只保留第一页里至少 8 条对得上容器名的查询，再用 offset 翻页（cursor 会把第一页排乱）。容器名要对上特异词 Jaccard 0.75，或者容器里出现会刊简称并且会名里的特异词都在。`query:` 后面的 reported 是词袋总数，不是该会的论文数。排序一旦漂移就停止，所以会议条数是对得上的那一段，不是会的全部论文。

## 2026-10-24 抓完的数

`collected/ccf_works.jsonl` 有 582,424 行、582,399 个不同 DOI。`collected/acl_works.jsonl` 有 56,548 行，和 Crossref member 1643 的报告数相同。两边重叠 1,637 个 DOI。合在一起是 637,310 个不同 DOI。另有 25 行 CCF 记录没有 DOI。

目录 645 条、634 个不同简称，每个简称都进了 `ccf_venue_log.txt`。172 次没有找到足够准的 Crossref 查询，记为 miss。其中 A 类有：FAST、USENIX ATC、SIGCOMM、NSDI、EUROCRYPT、S&P、CRYPTO、USENIX Security、POPL、OOPSLA、OSDI、FM、SIGMOD、VLDB、SODA、CAV、ACM MM、SIGGRAPH、VR、IEEE VIS、AAAI、ICML、Proc. IEEE、WINE。Theory of Cryptography Conference 和已经抓过的 IEEE Transactions on Cloud Computing 共用简称 TCC，续跑时被跳过，日志里补了一条 B 类 miss。

14 个会刊碰到 OpenAlex/Crossref 的页数上限，停在 8,000 条或提前停：HPCC 1865/2278，IOT 8000/19501，Information Sciences 8000/10083，JASA 8000/18375，MTA 8000/10052，ISMAR 1637/2184，EAAI 8000/11288，ESWA 8000/18876，KBS 8000/10084，NCA 8000/9035，Neurocomputing 8000/14791，TGARS 8000/16263，TITS 8000/9828，TII 8000/8682。TMC 的 ISSN 结果是 3633/5033，没有再标成截断。JMLR 的 ISSN 1532-4435 在这个日期窗里返回 0 条。

IEEE 的 2,945,079 条没有下载。SCI 和 EI 没有公开的全库接口，没有收进这份清单。

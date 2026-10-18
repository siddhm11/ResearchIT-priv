# arXiv HTML test fixtures

Six arXiv HTML pages, trimmed with `scripts/make_arxiv_fixture.py` to the sections
listed, used by `tests/test_arxiv_html.py` and the paper-source and Ask tests.
Each paper is licensed CC BY 4.0, checked on its arXiv page on 2026-10-17, which
allows redistribution with credit. Only ever add pages under CC BY, CC BY-SA or
CC0, and credit them here.

`<id>.expected.json` holds what the parser should extract from each page.
Regenerate after a deliberate parser change with
`pytest tests/test_arxiv_html.py --update-golden`, then review the diff.

| arXiv id | Title | Authors | Sections kept |
| --- | --- | --- | --- |
| 2609.31507 | [SatNav: A Scalable Benchmark for Long-Horizon UAV Vision-Language Navigation from Satellite Imagery](https://arxiv.org/abs/2609.31507) | Jiajun Jiang, Chunliang Hua, Zichun Chen, Yanxing Wu, Zeyuan Yang, Jie Song, Xiao Hu | S1,S3,S4,S6,S7 |
| 2609.20888 | [Elastic Threshold Attention: Learned Contextual Sparsity for Long-Context Decoding](https://arxiv.org/abs/2609.20888) | Themistoklis Haris, Henry Li, Maryam Karimzadehgan | S1,A3,S5 |
| 2609.27176 | [LeakScale: Estimating the Causal Effect of Benchmark Exposure](https://arxiv.org/abs/2609.27176) | Divyansh Singh | S1,S3,S4,S5,S6 |
| 2609.29861 | [GPT-6-Astra Lights Up Embodied Navigation: Evaluation in Zero-Shot Vision-and-Language Navigation in Continuous Environments](https://arxiv.org/abs/2609.29861) | Guangzhao Dai, Qianru Sun, Qi Wu, Bin Zhu | S1,S2,S4,S6 |
| 2609.30356 | [AlphaEarth distinguishes cities but compresses urban variation](https://arxiv.org/abs/2609.30356) | Andrew Renninger | S1,S3,S4 |
| 2609.30670 | [TRACE: Temporal Audit and Condition-aware Evaluation of Streaming Video Understanding](https://arxiv.org/abs/2609.30670) | Yibo Ma, Qianqian Zhang, Peng Liu, Tiancheng Zhao | S1,S3,S6 |

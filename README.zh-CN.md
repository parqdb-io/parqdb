<div align="center">
  <img src="https://raw.githubusercontent.com/petrizhang/relify/main/assets/relify-header.svg" alt="Relify" width="760">
  <p>
    <a href="https://github.com/petrizhang/relify/blob/main/README.md">English</a> |
    <a href="https://github.com/petrizhang/relify/blob/main/README.zh-CN.md">中文</a>
  </p>
  <p>
    <strong>为你已经在使用的 SQL 引擎提供开放向量索引。</strong>
  </p>
  <p>
    <a href="https://pypi.org/project/relify/"><img alt="PyPI" src="https://img.shields.io/pypi/v/relify.svg"></a>
    <a href="https://github.com/petrizhang/relify/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/petrizhang/relify/actions/workflows/ci.yml/badge.svg?branch=main"></a>
    <a href="https://github.com/petrizhang/relify/blob/main/pyproject.toml"><img alt="Python 3.11-3.14" src="https://img.shields.io/badge/python-3.11--3.14-blue.svg"></a>
    <a href="https://github.com/petrizhang/relify/blob/main/Cargo.toml"><img alt="Rust 1.96" src="https://img.shields.io/badge/rust-1.96-orange.svg"></a>
    <a href="https://github.com/petrizhang/relify/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT%20AND%20Apache--2.0-green.svg"></a>
  </p>
  <p>
    <a href="#快速开始">快速开始</a> |
    <a href="#为什么选择-relify">为什么选择 Relify</a> |
    <a href="#计算引擎">计算引擎</a> |
    <a href="#文档">文档</a>
  </p>
</div>

---

Relify 是一个使用 Python 和 Rust 开发的开源库，可通过你已经在使用的
计算引擎为湖仓数据构建向量索引并执行检索。它将向量索引存储为开放格式的
Parquet 数据集或 Iceberg 表，使 DataFusion、StarRocks 和 Spark 能够直接
通过 SQL 查询同一份索引，而源数据始终保留在原处。

如果 Relify 对你有帮助，欢迎点一个 Star，让更多人发现这个项目。

## 快速开始

Relify 支持 Linux x86_64 和 macOS arm64 上的标准 CPython 3.11 至 3.14。
安装默认的内嵌 DataFusion + Parquet 后端：

```bash
python -m pip install relify
```

Spark 和 StarRocks 是需要单独配置的可选集成，详见[计算引擎](#计算引擎)。

在一个新的工作目录中，使用包内置的数据集构建 IVF-Flat 索引，并执行带
过滤条件的向量查询：

```python
import relify

session = relify.connect("./relify-data")
session.register_parquet("documents", relify.datasets.uri("documents"))
documents = session.table("documents")

documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=relify.IVF(nlist=3),
)
documents.wait_for_index("documents_embedding")

query = (
    documents.search([0.2, 0.0], column="embedding")
    .where("tenant_id = 42 AND status = 'published'")
    .nprobes(3)
    .limit(3)
    .select(["document_id", "title", "category"])
)

print(session.collect(query).to_pylist())
```

向量检索的输出在 Relify 中仍然是一个关系，而不是一次执行后便终止的服务
调用。因此查询可以保持惰性，注册为 DataFusion 视图，再在同一执行上下文
中继续使用 SQL 分析：

```python
session.register_parquet(
    "document_stats",
    relify.datasets.uri("document_stats"),
)
session.register_view("vector_hits", session.to_dataframe(query))

summary = session.sql("""
    SELECT
        h.category,
        COUNT(*) AS matches,
        AVG(h._distance) AS avg_distance,
        MAX(s.popularity) AS max_popularity
    FROM vector_hits AS h
    JOIN document_stats AS s USING (document_id)
    GROUP BY h.category
    ORDER BY h.category
""")
print(summary.to_pydict())
```

包内置的数据集使上述示例可以独立运行。[入门指南](https://github.com/petrizhang/relify/blob/main/docs/getting-started.md)
介绍了持久化表、已有索引、查询计划检查和源表 Schema 要求。

## 为什么选择 Relify

- **无需将数据 ETL 到向量数据库。** 源向量保留在已有的湖仓表中，Relify 只
  写入索引数据和元数据。
- **一份开放向量索引。** IVF 中心点和倒排列表都是普通关系数据，以
  Parquet 数据集或 Iceberg 表发布，而不是某个引擎私有的二进制产物。
- **跨计算引擎使用。** 不同引擎的后端共享同一套索引模型和查询契约，无需
  为每个运行时维护单独的索引副本。
- **SQL 原生执行。** 聚类裁剪、源表过滤、Join、距离计算和 Top-K 都保留在
  宿主引擎的关系执行计划中。

## 计算引擎

| 引擎 | 模式 | 当前能力 | 状态 |
| --- | --- | --- | --- |
| DataFusion | 内嵌 | 在同一 Python 进程中构建和查询 Parquet 索引 | 已支持 |
| Spark Classic | 批处理 | 构建 Iceberg 索引；查询 Parquet 和 Iceberg | 实验性 |
| StarRocks | OLAP | 通过 Arrow Flight SQL 查询由 Spark 构建的 Iceberg 索引 | 实验性 |

DataFusion 是默认后端。Spark 和 StarRocks 位于 `relify.experimental` 下，
需要调用方自行管理引擎和 Catalog 配置。三者使用相同的查询模型和开放索引
元数据。

安装和配置方法参见[本地](https://github.com/petrizhang/relify/blob/main/docs/guides/local.md)、
[Spark](https://github.com/petrizhang/relify/blob/main/docs/guides/spark.md) 和
[StarRocks](https://github.com/petrizhang/relify/blob/main/docs/guides/starrocks.md)
指南。

## 文档

- [入门指南](https://github.com/petrizhang/relify/blob/main/docs/getting-started.md)
  和 [Python 示例](https://github.com/petrizhang/relify/tree/main/examples/python)
- [核心概念](https://github.com/petrizhang/relify/blob/main/docs/concepts.md)、
  [系统架构](https://github.com/petrizhang/relify/blob/main/docs/architecture.md)和
  [开放索引规范](https://github.com/petrizhang/relify/blob/main/spec/README.md)
- [Python API](https://github.com/petrizhang/relify/blob/main/docs/python-api.md)
  和[配置说明](https://github.com/petrizhang/relify/blob/main/docs/configuration.md)
- [当前限制](https://github.com/petrizhang/relify/blob/main/docs/limitations.md)、
  [故障排查](https://github.com/petrizhang/relify/blob/main/docs/troubleshooting.md)和
  [路线图](https://github.com/petrizhang/relify/blob/main/docs/roadmap.md)

## TEngineDB-V 与 Relify

[TEngineDB-V: An OLAP-Native Vector Search System for Large-k Workloads at
Tencent](https://arxiv.org/abs/2608.00650) 是腾讯面向大 K 向量检索的生产系统。
在百亿向量部署中，它通过与 TEngineDB 的深度集成，相比旧系统实现了最高
52 倍加速。

<p align="center">
  <img src="https://raw.githubusercontent.com/petrizhang/relify/main/assets/tenginedb-v-figure-7.png" alt="图 7：不同系统的延迟与召回率权衡" width="760">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/petrizhang/relify/main/assets/tenginedb-v-figure-13.png" alt="图 13：百亿规模生产环境性能" width="760">
</p>

Relify 借鉴了 TEngineDB-V 的理念，而非它的实现。Relify 围绕开放索引格式和
现有 SQL 引擎重新实现表原生向量检索，目标是在不依赖专有引擎的前提下达到
TEngineDB-V 级别的性能。

如果你在研究中使用 Relify，请引用我们的 VLDB 2026 Industry Track 论文：

```bibtex
@misc{wu2026tenginedbvolapnativevectorsearch,
  title         = {{TEngineDB-V}: An {OLAP}-Native Vector Search System for Large-$k$ Workloads at Tencent},
  author        = {Xufei Wu and Pengcheng Zhang and Yitong Song and Xiaobo Zhang and Anqi Liang and Kai Wang and Jijun Du and Yidi Xiong and Guangxu Cheng and Zhe Chen and Peng Chen and Guoliang Li and Xuanhe Zhou and Fan Wu},
  year          = {2026},
  eprint        = {2608.00650},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DB},
  url           = {https://arxiv.org/abs/2608.00650},
}
```

## 开发

Relify 下一阶段的方向正在公开讨论和设计。我们欢迎具体的使用场景、
Benchmark 结果、设计反馈和实现贡献：

- [围绕嵌入式向量湖仓收窄产品定位](https://github.com/petrizhang/relify/issues/9)
- [改进基于存储的 Parquet 检索](https://github.com/petrizhang/relify/issues/8)并
  [评估在线服务能力边界](https://github.com/petrizhang/relify/issues/11)
- [增加可扩展的索引类型框架](https://github.com/petrizhang/relify/issues/10)
- [设计存算分离架构](https://github.com/petrizhang/relify/issues/13)并
  [构建完整的 DuckLake 工作流](https://github.com/petrizhang/relify/issues/12)

如果你正在从事 RAG、Agent 轨迹存储、Parquet 性能优化或嵌入式湖仓系统，
欢迎在相关 Issue 中分享你的工作负载和需求。开始较大的改动前请先留言，以便
提前确定范围和接口。

Relify 使用 uv、Maturin、Cargo 和一层轻量的 Makefile 编排：

```bash
make sync
make develop
make check
```

质量门禁、Fixtures、Benchmark 和贡献规范详见
[CONTRIBUTING.md](https://github.com/petrizhang/relify/blob/main/CONTRIBUTING.md)。

## 许可证

Relify 的原创代码采用
[MIT License](https://github.com/petrizhang/relify/blob/main/LICENSE)。Wheel 中
包含采用 Apache-2.0 许可证的 DataFusion Python Binding；详情参见
[第三方声明](https://github.com/petrizhang/relify/blob/main/THIRD_PARTY_NOTICES.md)。

Relify 的开发受益于 LanceDB、DataFusion、DuckDB、StarRocks、Apache
Spark 和 Apache Iceberg 等项目，感谢这些项目的贡献者与社区。

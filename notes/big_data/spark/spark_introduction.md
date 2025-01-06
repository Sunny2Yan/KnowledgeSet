# Spark 介绍

Apache Spark: 用于大规模数据（large-scale data）处理的统一分析引擎
解决的问题：海量数据的计算，可以进行离线批处理以及实时流计算；

## 1. Spark 与 Hadoop 对比
|      | Hadoop            | Spark                   |
|------|-------------------|-------------------------|
| 类型   | 基础平台包括计算、存储、调度    | 纯计算工具（分布式）              |
| 场景   | 海量数据批处理（磁盘迭代）     | 海量数据批处理（内存迭代）、流处理       |
| 价格   | 对机器要求低            | 对内存有要求                  |
| 编程范式 | API较底层，算法适应性差     | RDD组成DAG有向无环图，API较顶层    |
| 数据存储结构 | 中间结果存储在HDFS磁盘上，延迟高 | RDD中间运算结果在内存中，延迟低       |
| 运行方式 | Task以进程方式维护，任务启动慢 | Task以线程方式维护，任务启动快，可批量创建 |

## 2. Spark 特点
1. 运行速度快：
    1）spark处理数据时，中间结果存储到内存中，MapReduce 与 HDFS 是通过磁盘交互；
    2）spark提供了丰富的api，复杂任务可以在一个spark程序中完成，MapReduce 只有两个算子，复杂任务会串联多个 MapReduce；
2. 使用简单：spark 支持 Java、Scala、Python、R和SQL 多种语言；
3. 通用性强：spark提供了Spark SQL、Spark Streaming、MLib（机器学习）、GraphX多个工具库；
4. 运行方式多：可以运行在Hadoop和Mesos上，也支持Standalone的独立运行模式，同时也可以运行在云Kubernetes上

## 3. Spark 框架模块
- Spark Core：spark的核心，以RDD为数据抽象，提供 Java、Scala、Python、R 和 SQL 的语言API；
- Spark SQL：结构化数据的处理模块，支持以 SQL 语言对数据进行处理；
- Spark Streaming：数据的流式计算功能；
- Spark MLib：机器学习计算，内置大量机器学习库和API；
- Spark GraphX：图计算，提供大量图计算API。

## 4. Spark 运行模式
- 本地模式（单机）：以一个独立的进程，通过内部多个线程来模拟整个spark运行时环境（常用于开发和测试）；
- Standalone（集群）：spark中的各个角色以独立进程的形式存在，并组成spark集群环境；
- Hadoop YARN模式（集群）：spark中的各个角色运行在YARN的容器内部，并组成spark集群环境；
- Kubernetes模式：spark中的各个角色运行在Kubernetes的容器内部，并组成spark集群环境。

## 5. Spark 架构角色：

1. YARN 角色：
- 资源层面：
    - ResourceManager: 集群资源管理
    - NodeManager: 单机资源管理
- 任务运行层面：
    - ApplicationMaster: 单个任务的管理
    - Task: 单个任务的计算

2. Spark 角色：

- 资源层面：
    - Master: 集群资源管理
    - Worker: 单机资源管理
- 任务运算层面：
    - Driver：单个任务的管理
    - Executor：单个任务的计算
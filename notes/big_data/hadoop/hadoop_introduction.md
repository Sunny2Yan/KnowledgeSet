# Hadoop 介绍

## 大数据生态
- 数据存储：Apache Hadoop-HDFS、Apache HBase(NoSQL KV)、Apache KUDU
- 数据计算：Apache MapReduce、Apache Hive(以SQL为开发语言，底层使用MapReduce)、Apache Spark、Apache Flink
- 数据传输：Apache Kafka、Apache Pulsar、Apache Flume、Apache Sqoop

1. Hadoop组件
- HDFS (Hadoop Distributed File System)：分布式文件存储组件
- MapReduce：分布式计算组件
- YARN：分布式资源调度组件

2. 分布式架构模式
- 去中心化：没有明确中心，基于特定规则进行同步协调；
- 中心化：某台服务器作为中心，统一调度其他节点；

一般是中心化模式（hadoop），即，主从模式(Master and Slaves)。

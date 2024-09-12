# HIVE

提供分布式 SQL 计算能力，及将 SQL 语句转换成 MapReduce，做分布式统计分析。

## 1. hive 架构

hive 组建：
1. Metadata Manager (data info)  
2. SQL Paser(sql分析，sql到mapreduce转换，提交mapreduce执行)

hive 组件：
```text
                            JDBC/ODBC
  command-line interface   hive thrift server    hive web interface
           |                     |                       |
 +-------------------------------------------------------------------+
 + hive driver:                                                      +
 +    Paser   Planner    Execution    Optimizer    MS Client    -----+----> Metastore
 +---------------------------|---------------------------------------+          |
               mapreduce    tez    spark                                      RDBMS
                             |
                        Hadoop YARN
                        Hadoop HDFS / Hbase
```

## 2. hive 部署

# 

## 大数据生态
- 数据存储：Apache Hadoop-HDFS、Apache HBase(NoSQL KV)、Apache KUDU
- 数据计算：Apache MapReduce、Apache Hive(以SQL为开发语言，底层使用MapReduce)、Apache Spark、Apache Flink
- 数据传输：Apache Kafka、Apache Pulsar、Apache Flume、Apache Sqoop

1. Hadoop组件
- HDFS (Hadoop Distributed File System)：分布式文件存储组件
- MapReduce：分布式计算组件
- YARN：分布式资源调度组件
- 
2. 分布式架构模式
- 去中心化：没有明确中心，基于特定规则进行同步协调；
- 中心化：某台服务器作为中心，统一调度其他节点；

一般是中心化模式（hadoop），即，主从模式(Master and Slaves)。

## HDFS
- NameNode：主角色（独立进程），管理整个HDFS系统及DataNode角色
- DataNode：从角色（独立进程），负责数据的读取
- SecondaryNameNode：主角色辅助角色（独立进程），帮助NameNode完成元数据整理

1. 部署HDFS
node1: NameNode, DataNode, SecondaryNameNode
node2: DataNode
node3: DataNode

Hadoop.apache.org -> Download -> Binary download
```bash
# 全部在node1 root下执行
tar -zxvf hadoop-xxx.tar.gz -C /export/server
ln -s /export/server/hadoop-xxx /export/server/hadoop

vim /export/server/hadoop/etc/hadoop/workers  # 记录从节点
   node1  # 删除loaclhost
   node2
   node3
   
vim /export/server/hadoop/etc/hadoop/hadoop-env.sh  # 配置环境变量
   export JAVA_HOME=/export/server/jdk
   export HADOOP_HOME=/export/server/hadoop
   export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop  # 配置文件目录
   export HADOOP_LOG_DIR=$HADOOP_HOME/logs
   
vim /export/server/hadoop/etc/hadoop/core-site.xml
   <configuration>
     <property>
       <name>fs.defaultFS</name>
       <value>hdfs://node1:8020</value>  # hdfs://为通讯协议，node1为NameNode，8020为通讯端口
     </property>

     <property>
       <name>io.file.buffer.size</name>
       <value>131072</value>  # io操作文件缓冲区大小131072bit
     </property>
   </configuration>
   
vim /export/server/hadoop/etc/hadoop/hdfs-site.xml
   <configuration>
     <property>
       <name>dfs.datanode.data.dir.perm</name>
       <value>700</value>  # 默认创建的文件权限 700，即 rwx------
     </property>
     <property>
       <name>dfs.namenode.name.dir</name>
       <value>/data/nn</value>  # 元数据的存储位置在node1节点下的 /data/nn/
     </property>
     <property>
       <name>dfs.namenode.hosts</name>
       <value>node1,node2,node3</value>  # NameNode允许哪些节点的DataNode连接
     </property>
     <property>
       <name>dfs.blocksize</name>
       <value>268435456</value>  # hdfs默认块大小 256MB
     </property>
     <property>
       <name>dfs.namenode.handler.count</name>
       <value>100</value>  # NameNode处理的并发线程数
     </property>
     <property>
       <name>dfs.datanode.data.dir</name>
       <value>/data/dn</value>  # DataNode的数据存储目录 /data/dn
     </property>
   </configuration>

mkdir -p /data/nn  # node1 上创建，文件名要与hdfs-site.xml中配置的保持一致
mkdir -p /data/dn  # node1, node2, node3 上创建

scp -r /export/server/hadoop-xxx node2:/export/server/
scp -r /export/server/hadoop-xxx node3:/export/server/

ln -s /export/server/hadoop-xxx /export/server/hadoop  # node2 上执行
ln -s /export/server/hadoop-xxx /export/server/hadoop  # node3 上执行

vim /etc/profile  # node1, node2, node3 都需要配置
   export HADOOP_HOME=/export/server/hadoop
   export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin  # :表示追加
   
# 为hadoop用户授权，上面都是root用户配置
chown -R hadoop:hadoop /data  # user:user group
chown -R hadoop:hadoop / export

# 初始化NameNode (node1)
su - hadoop
hadoop namenode -format

# 执行hadoop集群
start-dfs.sh  # 一键式启动集群，可以jps查看
stop-dfs.sh
```

`start-dfs.sh` -> 启动SecondaryNameNode -> 根据core-site.xml的记录启动NameNode -> 根据workers记录启动多个DataNode
http://node1:9870/  # 监控Hadoop
此时，可以进行快照，保存初始状态 （init 0 关机；init 6 重启）

2. 操作HDFS

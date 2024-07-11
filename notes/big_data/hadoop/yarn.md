

## 分布式计算
- 分散-汇总模式：数据分片，每台服务器各自计算，然后将结果汇总
- 中心调度-步骤执行模式：一个节点作为中心调度管理者，将任务划分为几个步骤，分别安排每台机器执行，然后汇总。

MapReduce（过时了）
分散-汇总模式的分布式计算框架，它提供两个接口：
- Map：提供分散功能，由服务器分布式对数据进行处理；
- Reduce：提供汇总（聚合）功能，将分布式计算结果汇总统计

原理：
1. 将数据分散，每台服务器处理部分数据
2. 将处理的结果放到另外一台服务器进行聚合

## YARN
YARN是一个分布式资源调度器（管控整个分布式集群的全部资源）。MapReduce是基于YARN运行的。

YARN 主从架构有两个角色：
- ResourceManager： 主角色（Master），整个集群的资源调度者，负责协调调度各个程序所需的资源；
- NodeManager：从角色（Slave），单个服务器的资源调度者，负责调度单个服务器上的资源（Container形式），提供给应用程序使用。

YARN 辅助架构有两个角色：
- 代理服务器（ProxyServer）：Web Application Proxy Web应用程序代理，减少通过YARN进行基于网络的攻击；
- 历史服务器（JobHistoryServer）：应用程序历史信息记录，产生日志（所有Container中的容器），以Web形式展示。

## YARN 部署
node1: ResourceManager, NodeManager, ProxyServer, JobHistoryServer
node2: NodeManager
node3: NodeManager

1. 配置 MapReduce 文件。在 `$HADOOP_HOME/etc/hadoop` 文件夹内修改：
```bash
# 1. 在 mapred-env.sh 文件中添加环境变量
export JAVA_HOME=/export/server/jdk  # jdk路径
export HADOOP_JOB_HISTORYSERVER_HEAPSIZE=1000  # 设置JobHistoryServer进程内存为1G
export HADOOP_MAPRED_ROOT_LOGGER=INFO,RFA  # 设置日志级别为INFO

# 2. 在 mapred-site.xml 中添加以下配置信息
<property>
   <name>mapreduce.framework.name</name>
   <value>yarn</value>
   <description>MapReduce的运行框架设置为YARN</description>
</property>
<property>
   <name>mapreduce.jobhistory.address</name>
   <value>node1:10020</value>
   <description>历史服务器通讯端口为node1的10020</description>
</property>
<property>
   <name>mapreduce.jobhistory.webapp.address</name>
   <value>node1:19888</value>
   <description>历史服务器web端口为node1的19888</description>
</property>
<property>
   <name>mapreduce.jobhistory.intermediate-done-dir</name>
   <value>/data/mr-history/tmp</value>
   <description>历史信息在HDFS的记录临时路径</description>
</property>
<property>
   <name>mapreduce.jobhistory.done-dir</name>
   <value>/data/mr-history/done</value>
   <description>历史信息在HDFS的记录路径</description>
</property>
<property>
   <name>yarn.app.mapreduce.am.env</name>
   <value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value>
   <description>MAPREDUCE_HOME 设置为 HADOOP_HOME</description>
</property>
<property>
   <name>mapreduce.map.env</name>
   <value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value>
   <description>MAPREDUCE_HOME 设置为 HADOOP_HOME</description>
</property>
<property>
   <name>mapreduce.reduce.env</name>
   <value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value>
   <description>MAPREDUCE_HOME 设置为 HADOOP_HOME</description>
</property>
```

2. 配置 YARN 文件。在 `$HADOOP_HOME/etc/hadoop` 文件夹内修改：
```bash
# 1. 在 yarn-env.sh 文件中添加环境变量
export JAVA_HOME=/export/server/jdk
export HADOOP_HOME=/export/server/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export HADOOP_LOG_DIR=$HADOOP_HOME/logs

# 2. 在 yarn-site.xml 中添加以下配置信息
<property>
   <name>yarn.resourcemanager.hostname</name>
   <value>node1</value>
   <description>ResourceManager设置在node1节点</description>
</property>
<property>
   <name>yarn.nodemanager.local-dirs</name>
   <value>/data/nm-local</value>
   <description>NodeManager中间数据本地存储路径</description>
</property>
<property>
   <name>yarn.nodemanager.log-dirs</name>
   <value>/data/nm-log</value>
   <description>NodeManager日志本地存储路径</description>
</property>
<property>
   <name>yarn.nodemanager.aux-services</name>
   <value>mapreduce_shuffle</value>
   <description>为MapReduce开启shuffle服务</description>
</property>
<property>
   <name>yarn.log.server.url</name>
   <value>http://node1:19888/jobhistory/logs</value>
   <description>历史服务器URL</description>
</property>
<property>
   <name>yarn.web-proxy.address</name>
   <value>node1:8089</value>
   <description>代理服务器主机和端口</description>
</property>
<property>
   <name>yarn.log-aggregation-enable</name>
   <value>true</value>
   <description>开启日志聚合</description>
</property>
<property>
   <name>yarn.nodemanager.remote-app-log-dir</name>
   <value>/tmp/logs</value>
   <description>程序日志HDFS的存储历经</description>
</property>
<property>
   <name>yarn.resourcemanager.scheduler.class</name>
   <value>org.apache.hadoop.yarn.server.resourcemanager.scheduler.fair.FairScheduler</value>
   <description>选择公平调度器</description>
</property>
```

3. 将配置好的文件发送到 node2, node3
```bash
scp mapred-env.sh mapred-site.xml yarn-env.sh yarn-site.xml node2:`pwd`/
scp mapred-env.sh mapred-site.xml yarn-env.sh yarn-site.xml node3:`pwd`/
```

## YARN 操作
1. yarn 启停操作
- 一键启停 YARN 集群
   `start-yarn.sh`：1) 读取yarn-site.xml的记录启动ResourceManager；2) 读取workers记录启动全部NodeManager。
   `stop-yarn.sh`：同上
   使用 $HADOOP_HOME/sbin/start-yarn.sh 和 $HADOOP_HOME/sbin/stop-yarn.sh 控制所有服务器的启停
- 单独启动 YARN 进程
   `yarn --daemon (start/stop) (resourcemanager/nodemanager/proxyserver)`：
   使用 $HADOOP_HOME/bin/yarn 控制脚本所在机器的进程的启停
- 历史服务器的启停
   `$HADOOP_HOME/bin/mapred --daemon (start/stop) historyserver`
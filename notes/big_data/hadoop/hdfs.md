# HDFS
HDFS系统包含三种角色：
- NameNode：主角色（独立进程），管理整个HDFS系统及DataNode角色
- DataNode：从角色（独立进程），负责数据的读取
- SecondaryNameNode：主角色辅助角色（独立进程），帮助NameNode完成元数据整理

## HDFS 部署
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

http://node1:9870/ (云部署需要使用外网IP)  # 监控Hadoop
此时，可以进行快照，保存初始状态 （init 0 关机；init 6 重启）

## HDFS 操作
1. hdfs 启停操作
- 一键启停 HDFS 集群
   `start-dfs.sh`：启动SecondaryNameNode -> 1) 读取core-site.xml的记录启动NameNode；2) 读取workers记录启动全部DataNode。
   `stop-dfs.sh`：关闭SecondaryNameNode -> 1) 读取core-site.xml的记录关闭NameNode；2) 读取workers记录关闭全部DataNode。
- 单独启动 HDFS 进程
   `hadoop-daemon.sh (status/start/stop) (namenode/secondarynamenode/datanode)`：
   使用 $HADOOP_HOME/sbin/hadoop-daemon.sh 控制脚本所在机器的进程的启停
   `hdfs --daemon (status/start/stop) (naenode/secondarynamenode/daatanode)`：
   使用 $HADOOP_HOME/bin/hdfs 控制脚本所在机器的进程的启停

2. hdfs 文件操作
文件系统（采用协议头）：
   linux: file:///  # 根目录
   hdfs: hdfs://namenode_ip:port/  # 根目录

```bash
hdfs dfs [generic option] / hadoop fs [generic option]  # 旧版本

hdfs dfs -put [-f] [-p] <linux_dir> <hdfs_dir>  # linux 上传到 hdfs
hdfs dfs -get [-f] [-p] <hdfs_dir> <linux_dir>  # hdfs 下载到 linux
   
# 只能删除和追加，不能修改
hdfs dfs -appendToFile <local_file> <hdfs_file>  # 本地文件追加到hdfs
hdfs dfs -rm [-r] [-skipTrash]  # 删除文件，hdfs有回收站，但默认关闭，需要在core-site.xml中配置开启

hdfs dfs -madir [-p] <path>  # 后面的命令同Linux相似
hdfs dfs -ls -R
hdfs dfs -cat <dir> | more
``` 

注：WEB UI也可以操作（Utilities -> Browse the file system）

3. hdfs 权限操作
hdfs超级用户：启动NameNode的用户 (linux 是root)

修改用户和组：`hdfs dfs -chown [-R] root:root /xxx.txt`。（注： user:user_group）
修改权限：`hdfs dfs -chmod [-R] 777 /xxx.txt`

## HDFS 客户端
### Jetbrains 中插件使用
在 IDE中安装 Big Data Tools 插件，并完成以下设置：
1. 解压Hadoop安装包到本地，如：`E:\\hadoop-3.3.4`
2. 设置环境变量 `$HADOOP_HOME` 指向 `E:\\hadoop-3.3.4`
3. 下载 `hadoop.dll` 和 `winutilsexe`
4. 将 `hadoop.dll` 和 `winutilsexe` 放入 `HADOOP_HOME/bin` 下

### 将 HDFS 挂在到本地
HDFS 提供了基于 NFS(Network File System) 的插件，使其能够挂载到本地，以供上传、下载、删除、追加操作。

1. 在node1的 `core-site.xml` 中追加
```bash
<property>
   <name>hadoop.proxyuser.hadoop.groups</name>
   <value>*</value>  # 允许hadoop用户代理任何其他用户组
</property>

<property>
   <name>hadoop.proxyuser.hadoop.hosts</name>
   <value>*</value>  # 允许代理任意服务器的请求
</property>
```

2. 在node1的 `hdfs-site.xml` 中追加
```bash
<property>
   <name>nfs.superuser</name
   <value>hadoop</value>   # NFS使用hadoop超级用户操作HDFS系统
</property>
<property>
   <name>nfs.dump.dir</name>
   <value>/tmp/.hdfs-nfs</value>  # 临时目录，在Linux本地
</property>
<property>
   <name>nfs.exports.allowed.hosts</name>
   <value>192.168.88.1 rw</value>  # 本机 VMnet8 的地址， rw允许读写
</property>
```

3. 将配置好的 `core-site.xml` 和 `hdfs-site.xml` 拷贝到 node2, node3
4. 重启 Hadoop HDFS 集群：`stop-dfs.sh`, `start-dfs.sh`
5. 停止系统的 NFS 相关进程
   `systenctl stop nfs`, `systemctl disable nfs`, `yum remove -y rpcbind`(移除系统自带的rpcbind)
6. root 启动 portmap(HDFS自带的功能)：`hdfs --daemon start portmap`
7. hadoop用户启动 nfs(HDFS自带的nfs功能)：`hdfs --daemon start nfs3`

检测：在node2, node3 上执行 `showmount -e node1`

程序 -> 启用或关闭Windows功能 -> NFS服务
CMD -> `net use X: \\192.168.88.101\!`

## HDFS 存储原理
将文件分成 N 个部分，分别存储到 N 个节点上，对于每个节点上的文件又划分成 256MB 的 Block。

为了安全常将每个 Block 复制几个副本到其他服务器。

- 查看文件副本数：`hdfs fsck /tmp/ [-files [-blocks [-locations]]]`

- 文件方式修改副本数：
```bash
vim hdfs-site.xml  # 每个节点都需要

<property>
   <name>dfs.replication</name>
   <value>3</value>  # 每个block存储3份(已经默认是3)
</property>
```
- 命令方式修改副本数：
   `hdfs dfs -D dfs.replication=2 -put test.txt /temp/`：上传时修改，`-D` 表示自定义配置项
   `hdfs dfs -setrep [-R] 2 test.txt`: 对已经存在的文件修改


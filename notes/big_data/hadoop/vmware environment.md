# 环境搭建

## VMware 环境
1. 设置 VMnet8 虚拟网卡 (88 网段是约定俗成)
   编辑 -> 虚拟网络编辑器 -> 更改设置 -> VMnet8 -> 子网IP（修改）-> NAT设置 -> 网关IP（修改）-> 确定
   - 网段（子网IP）：192.168.88.0
   - 网关（网关IP）：192.168.88.2

2. 创建虚拟机
   创建新的虚拟机 -> ..(稍后安装操作系统).. -> (默认即可)
   管理 -> 克隆 -> 创建完整克隆（node1. node2, node3）
   （node1 为主节点需要内存调大一点）

3. 虚拟机系统设置（Ubuntu）
   ```bash
   # 1.修改主机名
   su - root  # (ubuntu root用户默认锁定：sudo passwd root 即可)
   hostnamectl set-hostname node1
   
   # 2.修改IP地址
   vim /etc/netplan/01-network-manager-all.yaml 
   (centos: /etc/sysconfig/network-scripts/ifcfg-nes33)
   
   network:
   version: 2
   renderer: NetworkManager
   ethernets:
     ens33:   # 网卡名称
       dhcp4: false     # 关闭dhcp
       dhcp6: false
       addresses: [192.168.88.101/24]  # 静态ip
       routes:
         - to: default
           via: 192.168.88.2     # 网关
       nameservers:
         addresses: [192.168.88.2, ] # dns
   
   chmod 0600 01-network-manager-all.yaml
   
   # 3.重启网卡
   netplan apply (centos: systemctl restart network)
   
   # 4.配置主机映射
   vim C:\Windows\System32\drivers\etc\hosts
   192.168.88.101 node1
   192.168.88.102 node2
   192.168.88.103 node3
   
   vim /etc/hosts  # 三台虚拟机
   192.168.88.101 node1
   192.168.88.102 node2
   192.168.88.103 node3
   
   # 5.SSH远程连接
   # 安装SSH服务: apt-get upgrade | apt-get install openssh-server
   ssh -p 22 username@192.168.88.101
   
   # 6.配置root用户SSH免密登录 (每个虚拟机都需要执行)
   ssh-keygen -t rsa -b 4096
   ssh-copy-id node1  # 对node1免密
   ssh-copy-id node2
   ssh-copy-id node3
   
   # 注：一般会报错，解决如下：
   vim /etc/ssh/sshd_config 
     PermitRootLogin yes 
   /etc/init.d/ssh restart
   
   # 7.创建hadoop用户，并配置SSH免密登录
   # useradd后可能不显示用户名 usermod -s /bin/bash username
   useradd -m hadoop  # userdel  -m在home创建目录
   passwd hadoop
   
   ssh-keygen -t rsa -b 4096
   ssh-copy-id node1
   ssh-copy-id node2
   ssh-copy-id node3
   
   # 8.JDK环境部署 （下载 jdk.tar.gz）
   mkdir -p /export/server  # 之后存放软件的地方
   tar -zxvf jdk...tar.gz -C /export/server
   ln -s /export/server/jdk... /export/server/jdk  # 简化名字
   
   vim /etc/profile
   export JAVA_HOME=/export/server/jdk
   export PATH=$PATH:$JAVA_HOME/bin
   source /etc/profile
   
   rm -f /usr/bin/java  # 删除系统自带的java
   ln -s /export/server/jdk/bin/java /usr/bin/java
   
   # 9.关闭防火墙、SELinux （root）
   systemctl stop firewalld
   systemctl disable firewalld  # 关闭开机自启
   
   vim /etc/sysconfig/selinux
   SELINUX=disabled  # 注：写错系统将无法启动
   
   # 10.时间同步
   apt install ntpdate (yum install -y ntp)
   rm -f /etc/localtime | ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime  # 更新时区
   ntpdate -u ntp.aliyun.com  # 同步时间
   systemctl start ntpd
   systemctl enable ntpd
   ```

4. 保存初始设置
   快照 -> 拍摄快照

## 云平台环境
- IaaS服务 (Infrastructure as a Service): 基础设施服务
   IT基础设施：公网IP、带宽、VPC(virtual private cloud)、云服务(CPU, 内存, 磁盘...) ...
- PaaS服务 (Platform as a Service): 平台服务
   平台服务：小程序开发平台、机器学习平台 ...
- SaaS服务 (Software as a Service): 软件服务
   软件服务：云上数据库、云盘、域名托管服务 ...

设施层 -> 平台层 -> 应用层

### 创建阿里云环境 （其他云同理）
1. 配置私人局域网 (VPC)
aliyun.com -> 登录 -> 控制台 -> 专有网络 VPC -> 专有网络 -> (选择区域) -> 创建专有网络：
   IPv4网段：192.168.0.0/16  (16表示后两位可变，24表示后一位可变)
   IPv6网段：不分配
   交换机(子网)：选择机房 192.168.88.0/24

2. 配置安全组 (虚拟流量防火墙)
aliyun.com -> 登录 -> 控制台 -> 云服务器 ECS -> 安全组 -> 创建安全组：
   网络：上面创建的VPC
   配置规则：入方向只允许自己访问

3. 创建云服务器
aliyun.com -> 登录 -> 控制台 -> 云服务器 ECS -> 创建我的ECS
   node1: 1 core, 4GB, 20GB -> 192.168.88.101
   node2: 1 core, 2GB, 20GB -> 192.168.88.102
   node2: 1 core, 2GB, 20GB -> 192.168.88.103

4. 服务器配置
   主机名映射，root / Hadoop的SSH免密登录，JDK环境
   关闭firewall和SELinux (默认已经关闭)，设置时区同步 (默认已经设置好)
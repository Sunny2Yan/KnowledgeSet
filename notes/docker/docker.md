# Docker

- Linux 容器: 不是模拟一个完整的操作系统，而是对进程进行隔离。或者说，在正常进程的外面套了一个保护层。对于容器里面的进程来说，它接触到的各种资源都是虚拟的，从而实现与底层系统的隔离。 
- Docker: 属于 Linux 容器的一种封装，提供简单易用的容器使用接口。Docker 将应用程序与该程序的依赖，打包在一个文件里面。运行这个文件，就会生成一个虚拟容器。它是对进程进行隔离。

## 安装
1. Windows 安装
Windows：“启用或关闭Windows功能” --> 勾选 Hyper-V 和容器复选框 --> 下载 docker 安装包直接安装。
注：出现 “WSL 2 installation is in...” 报错时，需要安装 linux 内核更新包。

2. Linux brash 脚本安装：
`curl -flSL get.docker.com -o get_docker.sh`   下载文件安装包
`sudo sh get_docker.sh --mirror Aliyun`   镜像下载

3. 添加加速镜像
setting --> Docker Engine 添加
```
"registry-mirrors": ["https://docker.mirrors.ustc.edu.cn/",
                     "https://hub-mirror.c.163.com/",
                     "https://reg-mirror.qiniu.com"],
```
检查是否生效：`docker info`

## 4. docker 架构

Client：`docker build`，`docker pull`，`docker run`。。。

Docker Host：docker daemon(服务器组件)，images(文件)，Container(image的实例化)

Registry：

```bash
-it  # 以交互模式进入镜像
-t   # 为镜像命名
-d   # 以后台方式启动容器
--name  # 为容器命名
-p   # 端口映射
```

## 操作
```bash
systemctl status docker  # 查看状态  ubuntu: service docker status
systemctl start docker  # 启动
systemctl stop docker  #停止
systemctl restart docker  #重启

# 2.
docker info  # 查看引擎版本
# 注权限不足时，添加用户权限：
sudo gpasswd - a username docker  # 将用户username加入docker 组
newgrip docker  # 更新docker 组

# 3.
systemctl enable docker  # 设置开机自启动

# 4.建立docker组，并使用root用户（安全性）
sudo groupadd docker  # 建组
sudo usermod - aG docker $USER
```
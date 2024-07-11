# Docker Images

Dockerfile 是镜像描述文件，定义如何构建 docker 镜像。

## 构造 Dockerfile 镜像文件
1. Dockerfile 指令
```bash
FROM        # 指定base镜像
MAINTAINER  # 设置镜像作者
WORKDIR     # 后续命令运行的目录
RUN         # 容器中运行指定的指令，每次RUN都是一层容器，目录不共用（需要指定WORKDIR）
COPY        # 从build context复制到镜像
ADD         # 同copy
ENV         # 设置环境变量
EXPOSE      # 指定容器中的进程监听某个端口
VOLUME      # 将文件 or 目录声明为volume
CMD         # 容器启动时运行的指令
ENTRYPOINT  # 设置容器启动时运行的命令，如 ["./your_executable_or_script.sh"]
```

2. 举例
```dockerfile
# 1.base镜像：不依赖于其他镜像(从scratch构建)，且其他镜像可以扩展它。
FROM scratch
# ADD 自动解压生成镜像文件
ADD centos-7-docker.tar.gz /
CMD ["/bin/bash"]

# 2.添加镜像层
FROM debain
RUN apt-get install emacs
# CMD: 容器启动时运行bash
CMD ["/bin/bash"]
```

注：Dockerfile的指令每执行一次都会在docker上新建一层。所以过多无意义的层，会造成镜像膨胀过大。例如：

```dockerfile
FROM centos
RUN yum -y install wget
RUN wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz"
RUN tar -xvf redis.tar.gz

# 以上执行会创建 3 层镜像。可以使用 && 符号连接命令，只会创建 1 层镜像：
FROM centos
RUN yum -y install wget && \
    wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz" && \
    tar -xvf redis.tar.gz
```
**注：镜像层都是只读的，只有容器层是可写的。**

## docker image 命令
```bash
# 1.基本命令
docker search xxx   # 从Docker Hub搜索镜像
docker pull xxx     # 下载镜像
docker images xxx   # 查看镜像列表
docker run xxx      # 运行镜像
docker run -it xxx  # 运行并进入镜像，-it以交互模式进入镜像
docker rmi xxx      # 删除镜像

# 2.构建镜像
docker history         # 镜像构建历史
# 方法一：
docker commit xxx yyy  # 将容器xxx打包成镜像yyy(需要在新窗口查看运行容器，再打包)
# 方法二：
新建 Dockerfile 并在当前目录下执行
docker build -t xxx .  # -t xxx 为新镜像命名，.指当前目录

# 3.分发镜像
docker tag xxx:v1      # 给镜像打标签
docker push xxx        # 推送到xxx仓库
```

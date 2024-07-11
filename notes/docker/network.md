## Docker Network

```bash
docker network ls                    # 查看网络
docker network inspect xxx           # 查看xxx网络的具体信息
docker run -it --network=xxx ubuntu  # 以xxx网卡进入容器(挂载到同一网卡的容器可以互通)
docker network connect xxx container_id    # 在容器中添加一个xxx的网卡(新窗口docker ps 再添加)
```

1. docker 内置网络
```bash 
# 1.none 网络
--network=none     # 网卡为空

# 2.host 网络
--network=host     # 容器网络配置同host

# 3.bridge 网络 (default)
```

2. user-defined 网络
docker 提供三种 user-defined 网络驱动：beidge，overly，macvlan。后两种用于跨主机网络

```bash
# 1.bridge
docker network create --driver bridge xxx  # 创建名为xxx的网络
  --subnet   # 指定子网(范围)
  --gateway  # 指定网关
  --ip       # 指定静态IP(子网中的一个值)
  
eg: docker network create --driver bridge --subnet 172.22.16.0/24 --gateway 172.22.16.1 test
    docker run -it --network=test --ip 172.22.16.4 test_1  # 只有指定subnet的网络才能指定IP
```

3. 容器通信

```bash
# 1.容器间通信
  # 1) IP通信：拥有相同的网卡 docker network connect
  # 2) DNS Server：直接通过容器名通信(必须再user-defined中，内置网络不行)
docker run -it --network=test --name=xxx ubuntu docker run -it --network=test --name=yyy ubuntu

  # 3) joined 容器：使多个容器共享一个网络栈
docker run -it --network=container:xxx_name ubuntu  # ubuntu容器同xxx容器共享网络

# 2.容器与外界通信
  # 1) 容器访问外部世界
  # 2) 外部世界访问容器(端口映射)
docker run -it -p 80 ubuntu       # 将docker容器的80端口映射到外部动态端口
docker run -it -p 8080:80 ubuntu  # 将docker容器的80端口映射到外部8080d
docker ps / docker port xxx_name  # 查看端口    
```

## Docker Storage
docker volume: docker 的一种存储机制，是一个目录或文件。volume 数据可以被永久保存，即使使用它的容器已被销毁。
docker 提供两种类型的 volume：bind mount，docker managed volume。

1. bind mount
```bash
# 1.bind mount: 将host上已存在的目录或文件mount到容器
docker run -d -p 80:80 -v ~/test:/home/xxx:ro ubuntu  # -v <host path>:<container path>
    
# 2.docker managed volume: 不需要指明mount源，指明mount point就行
docker run -d -p 80:80 -v /home/xxx ubuntu  # -v <container path>
```

2. docker managed volume
```bash
# 容器与host数据共享
docker cp ~/xxx ubuntu_name:/usr/local/apache

# volume container (数据在host)
# 1.创建共享容器(只提供数据，不运行，create)
docker create --name vc_data \
    -v ~/htdocs:/usr/local/apache  # bind mount 存放web server静态文件
    -v /other/useful/tools \       # docker managed volume 存放实用工具
    ubantu
# 2.共享容器(--volume-from)
docker run --name xxx -d -p 80 --volumes-from vc_data ubantu

# data-packed volume container (数据在container)
FROM ubuntu:latest
ADD htdocs /usr/local/apache
VOLUME /usr/local/apache

docker build -t datapacked .                         # 将dockerfile生成镜像
docker create --name vc_data datapacked              # 创建volume container
docker run -d -p 80:80 --volume-from vc_data ubuntu  # 创建容器
    
# 生命周期管理
1. volume实际上是host中的目录和文件 (/myregistry)
2. 迁移: docker run -d -p 80:80 -v /myregistry:/var/lib/registry registry:latest
3. 删除: docker volume rm xxx
4. 批量删除孤儿volume: docker volume rm $(docker volume ls -q)
```

## Multi-Host 管理
Docker Machine 批量安装和配置 docker host。（需要下载 docker machine）

```bash
# 1.测试docker machine
docker-machine version

# 2.创建machine
docker-machine ls  # 查看当前可用machine
ssh-copy-id 192.168.1.1  # 将ssh-key复制到192.168.1.1，使能够无密码登录远程主机
docker-machine create --driver generic --generic-ip-address=192.168.1.1 host1  # 创建host1

# 3.管理machine
docker -H tcp://192.168.1.1:2376 ps  # docker远程连接
docker-machine env host1  # docker machine远程连接，简化上述命令
 eval $(docker-machine env host1)

docker-machine upgrade  # 更新machine的docker
    # eg: docker-machine upgrade host1 host2
docker-machine config   # 查看machine的docker daemon配置
    # eg: docker-machine config host1
docker-machine scp      # 不同machine之间复制文件
    # eg: docker-machine scp host1:/tmp/a host2:/tmp/b
stop/start/restart      # 针对machine的操作系统，而不是docker daemon
```


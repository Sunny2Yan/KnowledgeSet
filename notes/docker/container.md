# Docker Container

## Container 常用命令

```bash
docker ps              # 查看当前运行的容器
docker container ls    # 查看当前运行的容器
docker run             # 启动容器 = docker create + docker start
docker attach          # 进入容器，不会启动新的进程
docker exec -it        # 进入容器，可以启动新的进程
docker logs            # 查看日志

docker start       # 保留容器的第一次启动时的所有参数
docker restart     # 重启容器 = docker stop + docker start
docker stop        # 停止运行的容器
docker kill        # 快速停止容器

docker pause      # 暂停容器，不占用资源
docker unpause    # 恢复暂停

docker rm         # 删除容器
```

## 资源限制参数

```bash
# 1.内存限额
-m / --memory      # 设置内存使用额度
--memory-swap      # 设置内存+swap使用额度，若不指定，default = 2 * memory
eg: docker run -m 200 --memory-swap 300 ubuntu  # memory=200, swap=100
    
--vm n             # 启动n个内存工作线程
--vm-bytes 280M    # 每个线程分配280内存
eg: docker run -m 200 --memory-swap 300 ubuntu ubuntu --vm 1 --vm-bytes 300M
    
# 2.CPU限额
-c / --cpu-shares  # 设置cpu权重，相对权重
eg: docker run --name "xxx" -c 1024 ubuntu docker run --name "yyy" -c 512 ubuntu
    # 只有在资源紧张时才会按权重分配
--cpu              # 设置工作线程的数量

# 3.Block IO(磁盘读写) 带宽限额
--blkio-weight     # 设置磁盘读写优先级，同 --cpu-shares

# bps (byte per second)  每秒读写的数据量
# iops (io per second)   每秒IO的次数
--device-read-bps     # 限制读某个设备的bps
--device-write-bps    # 限制写某个设备的bps
--device-read-iops    # 限制读某个设备的iops
--device-write-iops   # 限制写某个设备的iops
eg: docker run -it --device-write-bps /dev/sda:30M ubuntu
```


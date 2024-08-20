安装
```bash
# Ubuntu/Debian
sudo apt install lsb-release curl gpg  # 如果在容器中安装需要执行此步骤

curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list

sudo apt-get update
sudo apt-get install redis
```

启动redis
```bash
systenctl status/start/stop/restart redis-server 
service redis-server status/start/stop/restart

# 默认只能本地访问redis，如需要在其他机器访问，则需要配置 /etc/redis/redis.conf 文件
# bind 127.0.0.1 -::1
requirepass 123456  # 配置密码
```


本机client 连接redis-server
redis-cli


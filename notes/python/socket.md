# 网络编程


## 1. 服务端
```python
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP 套接字
s.bind(('', 8989))  # TCP 绑定端口
s.listen()
while True:
    client, addr = s.accept()
    print("连接来自：", addr)
    msg = "显示服务器返回数据"
    while True:
        data = client.recv(1024)
        if data == 0:
            print("no data")
        else:
            print(data.decode('utf-8'))
            client.send(msg.encode)
    client.close()
s.close()
```


## 2. 客户端

```python
import socket

socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_client.connect(('10.4.143.178', 8989))

data = "我是要发送的数据"
socket.send(data.encode())
msg = socket_client.recv(1024)
print(msg.decode('utf-8'))
socket_client.close()
```
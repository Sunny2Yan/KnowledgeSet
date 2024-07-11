# TOML 文件格式
TOML (Tom's Obvious, Minimal Language) 是一种语义明显且易于阅读的最小化配置文件格式，被设计成可以无歧义地映射为哈希表。

```toml
# 1. 键名可以是裸露的，引号引起来的，或点分隔的。
title = "TOML 示例"  # 值有类型，必须加引号
physical.shape = "round"  # "physical": {"shape": "round"}

# 2. 多行字符串（加反斜杠\，不会产生换行符）
str3 = """\
       The quick brown \
       fox jumps over \
       the lazy dog.\
       """

# 3. 整数可以添加下划线_，增强可读性
x = 1_000
y = inf  # 正无穷

# 4. bool要小写
bool1 = false

# 5. [table] 就是一个hash-table，表名也可以使用点分隔
[server]  # [[server]] = {{"server": }}
host = "localhost"
port = 8080

[servers.alpha]  # <==> {"servers": {"alpha": {"ip": "10.0.0.1", "role": 前端}}}
ip = "10.0.0.1"
role = "前端"

```

## python 读写 `.toml` 文件

```python
# pip install toml
import toml

config = toml.load('config.toml')

host = config['server']['host']  # json的读取方法

x ={}
with open("xx.toml", 'w') as file:
    toml.dump(x, file)
```
# INI文件格式
`.ini` 是一种常见但非正式的配置文件格式。

```ini 
; ini文件分为各个section，每个section保存key-value
; 注意字符串没有引号，且等号不留空格
[database]
url=jdbc:mysql://localhost:3306/mydb
username=admin
password=secret

[server]
host=localhost
port=8080
```

## python 读写 `ini` 文件

```python
import configparser


config = configparser.ConfigParser() # 类实例化
config.read('xxx.ini')

# 1. 获取所有section
sections = config.sections()  # [sections_1, sections_2, ...]

# 2. 读取值（读取的都是 str）
value = config['select']['url']
value = config.get('select','url')  # getint(), getfloat(), getboolean()
value = config.items('select')  # [(k_1, v_1), (k_2, v_2), ...]

# 3. 写入ini文件
config.add_section('login')  # 先添加一个新的section
config.set('login','username','admin')  # 写入数据
config.set('login','password','123456') # 写入数据
config.write(open('xxx.ini', 'a'))      # 保存数据
```


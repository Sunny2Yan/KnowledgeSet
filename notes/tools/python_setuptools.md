# Python 打包分发

使用setuptools进行python打包

## 1. setup.py 介绍
使用 pycharm Tools -> Create setup.py 生成简易 setup.py 文件。

```python
from setuptools import setup

setup(
        name='demo',     # 包名字
        version='1.0',   # 包版本
        description='This is a test of the setup',   # 简单描述
        author='huoty',  # 作者
        author_email='sudohuoty@163.com',  # 作者邮箱
        url='https://www.konghy.com',      # 包的主页
        packages=['demo'],                 # 包
)
```

- packages: 对于复杂项目可以使用 `find_packages`。
   `find_packages(where='.', exclude=(), include=('*',))`
   where : 搜索路径；exclude: 排除哪些包；include: 包含哪些包
- 如果不用 find_packages，想要找到其他目录下的包可以使用 `package_dir={'my_package': 'src/my_package'}`



example
```python
from setuptools import setup, find_packages

install_requirements = []
with open('./requirements.txt', 'r', encoding='utf-8') as f1:
    for requirement in f1.readlines():
        install_requirements.append(requirement.strip('\n'))


setup(
    name='enn-benchmark',
    version='0.0.1',
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    install_requires=install_requirements,
    url='https://gitlab.enncloud.cn/FL-Engine/enn-benchmark',
    license='',
    author='enn team',
    author_email='',
    description='This is a LLM evaluation benchmark.'
)
```

## 2. setup.py 安装

```bash
python setup.py check  # 检查 setup.py 文件

python setup.py sdist  # 打包为 tar
python setup.py bdist_egg  # 打包为 egg
python setup.py bdist_wheel  # 打包为 wheel
cd dist | pip install ./xxx  # 安装

pip install -e .  # 直接安装
python setup.py install  # 直接安装
```
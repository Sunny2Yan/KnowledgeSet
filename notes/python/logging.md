# logging 模块

## 1. 分级日志记录
该方法下所有的 Python 模块都会参与日志输出，包括自己的日志消息和第三方模块的日志消息。

通过 `getLogger(__name__)` 创建一个模块级别的日志记录器，并使用该日志记录器来完成任何需要的日志记录。 
记录到模块级日志记录器的消息会被转发给更高级别模块的日志记录器的处理器，一直到最高层级的日志记录器既根日志记录器。
`basicConfig()` 提供了一种配置根日志记录器的快捷方式.
```python
import logging


logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(
        filename='mylog.log', 
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s : %(message)s')
    logger.info('Started')
```
此时，在别的文件下不需要调用，就可以直接使用

```python
import logging


logger = logging.getLogger(__name__)

logger.info("Finished")
```

## 2. 
- 记录器对象: `getLogger()`
- 格式器对象: `logging.Formatter()`
- 处理器对象: `Handler()`，通过 `Logger.addHandler()` 和 `Logger.removeHandler()` 从记录器对象中添加和删除处理器对象
- 过滤器对象: `logging.Filter()`，通过 `Logger.addFilter()` 和 `Logger.removeFilter()` 添加或移除记录器对象中的过滤器

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # 外部记录器的消息将不会输入到本记录器

console_handler = logging.StreamHandler()  # 创建 console handler
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('mylog.log')  # 创建 file handler
file_handler.setLevel(logging.INFO)

# 设置格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.info("message")
```
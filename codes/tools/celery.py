# -*- coding: utf-8 -*-
from celery import Celery
from celery.result import AsyncResult


# 1. config file
broker_url = "redis://localhost:6379/0"
# CELERY_RESULT_BACKEND = "db+mysql+pymysql://web_joinlearn:wjWTDoH9pSap@10.39.68.99/test"
result_backend = "redis://localhost:6379/1"

result_serializer = 'json'   # 结果序列化方案
result_expires = 60 * 60 * 24   # 任务过期时间
timezone = 'Asia/Shanghai'   # 时区配置
WORKER_PREFETCH_MULTIPLIER = 1
task_acks_late = True

imports = ('enn_benchmark.celery_worker.tasks', )


# 2. app file
app = Celery('enn_app')
app.config_from_object('config')


# 3. task file
@app.task
def test_task():
    return "Executed successfully!"


# 执行任务
res = test_task.delay()
task = AsyncResult(id=res.id)
print(task.status)
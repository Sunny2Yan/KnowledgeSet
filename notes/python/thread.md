# MultiProcess, Thread, Asyncio

## 1. Process
进程间的通信使用 queue

```python
import time
from multiprocessing import Process, Pool, Lock
from concurrent.futures import ProcessPoolExecutor

def work(i: int):
    print('调用第{}个进程'.format(i))

# 1. 直接创建进程
for i in range(10):  # 创建10个线程
    t = Process(target=work, args=(i,))
    t.start()
    t.join()

# 2. 进程池Pool批量启动子进程，避免 Process 在创建、销毁时的开销
with Pool(5) as p:
    # map()属于同步阻塞，map_async()异步非阻塞，较推荐
    res1 = p.map(work, [1, 2, 3, 4, 5])

# 3. concurrent.futures包实现
with ProcessPoolExecutor() as executor:
    res2 = executor.map(work, [6, 7, 8, 9, 10])
    
    future = executor.submit(work, 100, 200)    
    res3 = future.result()

# 4. 进程锁
def work2(name, lock):
    lock.acquire()  # 或使用上线下文 with lock:
    for i in range(5):
        print(name)
        time.sleep(1)
    lock.release()  # 不释放会造成死锁

lock = Lock()
for i in range(3):
    t = Process(target=work2, args=('task{}'.format(i), lock))
    t.start()
    t.join()
```

## 2. Thread
1. 当内存只够容纳 n 个线程时，需要通过信号量（Semaphore），用来保证多个线程不会互相冲突。
2. 当内存只够容纳 1 个线程时，其他线程必须等它结束，才能使用这一块内存，即需要添加互斥锁（Mutual exclusion，Mutex）防止多个线程同时读写某一块内存区域。

注：GIL（Global Interpreter Lock)  的存在让 Python 的多线程应用只能实现并发，而不能实现并行；多线程常用于 I/O 密集型。

```python
from threading import Thread, Lock
from multiprocessing.pool import ThreadPool

def work(i: int):
    print('调用第{}个线程'.format(i))

# 1. 直接创建线程 
for i in range(10):  # 创建10个线程
    t = Thread(target=work, args=(i,))
    t.start()
    t.join()

# 2. 创建线程池，避免 Thread 在创建、销毁时的开销
t = ThreadPool(5)
for i in range(10):
    t.apply_async(func=work, args=(i, ))
t.close()  # 使用with上下文管理则不需要
t.join()

# 3. 线程锁，防止线程间共享的数据被多个线程同时修改
share_var = 0

def change_it(n):
    global share_var
    share_var = share_var + n
    share_var = share_var - n

lock = Lock()
def work_1(n):
    for i in range(100000):
        lock.acquire()  # 获取锁
        try:
            change_it(n)  # 修改值
        finally:
            lock.release()  # 改完需要释放锁
```

## 3. Asyncio

```python
import time
import asyncio

async def visit_url(url, response_time):
    """访问 url"""
    await asyncio.sleep(response_time)
    return f"访问{url}, 已得到返回结果"

async def run_task():
    """收集子任务"""
    task_1 = visit_url('https://www.baidu.com', 2)
    task_2 = visit_url('https://www.bing.com', 3)

    task1 = asyncio.create_task(task_1)
    task2 = asyncio.create_task(task_2)

    await task1
    await task2

start_time = time.time()
asyncio.run(run_task())
print(f"消耗时间：{time.time() - start_time}")  # 3.s(发生并发)
```
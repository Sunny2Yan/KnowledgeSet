# C Python


```python
import ctypes
ll = ctypes.cdll.LoadLibrary
lib = ll("./testcc.so")
lib.foo(1, 3)
print('***finish***')
```
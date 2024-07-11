# YAML 文件格式

YAML 是专用来写配置文件的语言，非常简洁和强大，远比 JSON 格式方便。YAML 语言（发音 /ˈjæməl/ ）的设计目标，就是方便人类读写。其本质是一种通用的数据串行化格式。它以 .yml 后缀，基本语法规则如下：

- 大小写敏感， # 注释
- 使用缩进表示层级关系
- **缩进时不允许使用Tab键，只允许使用空格**。
- **缩进的空格数目不重要，只要相同层级的元素左侧对齐即可**

YAML 支持的数据结构有三种：
- 对象：键值对的集合，又称为映射（mapping）/ 哈希（hashes） / 字典（dictionary）
- 数组：一组按次序排列的值，又称为序列（sequence） / 列表（list）
- 纯量（scalars）：单个的、不可再分的值

## 1. 对象（字典）
字典使用冒号结构表示。

```yaml
animal: pets                   # {animal: 'pets'}
hash: {name: Steve, foo: bar}  # {hash: {name: 'Steve', foo: 'bar'}}    
```

## 2. 数组（列表）
列表是一组连词线开头的行。

```yaml
# 1. 基本表示
- Cat
- Dog
- Goldfish    # ['Cat', 'Dog', 'Goldfish']
# 2. 缩进一个空格表示子列表    
-
  - Cat
  - Dog
  - Goldfish  # [['Cat', 'Dog', 'Goldfish']]
```

### 1_2. 字典与列表组合
```yaml
animal: [Cat, Dog]  # {animal: ['Cat', 'Dog']}
languages:
  - Ruby
  - Perl
  - Python
websites:
  YAML: yaml.org 
  Ruby: ruby-lang.org 
  Python: python.org 
  Perl: use.perl.org
# {languages: ['Ruby', 'Perl', 'Python'], 
#  websites: {YAML: 'yaml.org', 
#             Ruby: 'ruby-lang.org', 
#             Python: 'python.org', 
#             Perl: 'use.perl.org'}}
```

## 3. 纯量（常量）
纯量是最基本的、不可再分的值。如：字符串、布尔值、整数、浮点数、Null、时间、日期

```yaml
# 1.数值
number: 12.30     # {number: 12.30}
 
# 2.布尔型(true / false)
isSet: true       # {isSet: true}

# 3.空值  (~)
parent: ~         # {parent: null}

# 4.时间  (ISO8601格式)
iso8601: 2001-12-14t21:59:43.10-05:00  
# {iso8601: new Date('2001-12-14t21:59:43.10-05:00')}

# 5.日期  (iso8601格式)
date: 1976-07-31  # {date: new Date('1976-07-31')}

# 6.强制转换类型 (!!)
e: !!str 123
f: !!str true     # {e: '123', f: 'true'}  
```

### 3.1 字符串
字符串是最常见，也是最复杂的一种数据类型。默认不使用引号表示

```yaml
# 1.基础
str_1: 这是一行字符串     # {str_1: '这是一行字符串'}
str_2: ''              # {str_2: ''(空字符串)}
	
# 2.若包含空格或特殊字符，需放到引号中
str_3: '内容: 字符串'    # {str: '内容: 字符串' }
	
# 3. 双引号不会对特殊字符转义
s1: '内容\n字符串'
s2: "内容\n字符串"     # {s1: '内容\\n字符串', s2: '内容\n字符串'}
	
# 4.单引号之中如果还有单引号，必须连续使用两个单引号转义
str_4: 'labor''s day'  # {str_4: 'labor\'s day'}

# 5.字符串可以写成多行，从第二行开始，必须有一个单空格缩进。换行符会被转为空格
str_5: 这是一段 
  多行
  字符串              # {str_5: '这是一段 多行 字符串'}
	  
# 6.多行字符串可以使用|保留换行符，也可以使用>折叠换行
this: |
  Foo
  Bar 
that: >
  Foo
  Bar                
# {this: 'Foo\nBar\n', that: 'Foo Bar\n' }

# 7.+表示保留文字块末尾的换行，-表示删除字符串末尾的换行：
s3: | 
  Foo
s4: |+
  Foo
s5: |-
  Foo               # {s1: 'Foo\n', s2: 'Foo\n\n\n', s3: 'Foo'}

# 8.字符串之中可以插入 HTML 标记：
message: | 
  <p style="color: red">
    段落
  </p>              # {message: '\n<p style="color: red">\n  段落\n</p>\n'}
```

## 4. 引用
锚点 `&` 和别名 `*`，可以用来引用。`&` 用来建立锚点，`<<` 表示合并到当前数据，`*` 用来引用锚点。

```yaml
defaults: &defaults
  adapter:  postgres
  host:     localhost

development:
  database: myapp_development
  <<: *defaults

test:
  database: myapp_test
  <<: *defaults

# defaults:
#   adapter:  postgres
#   host:     localhost

# development:
#   database: myapp_development
#   adapter:  postgres
#   host:     localhost

# test:
#   database: myapp_test
#   adapter:  postgres
#   host:     localhost

- &showell Steve 
- Clark 
- Brian 
- Oren 
- *showell  # ['Steve', 'Clark', 'Brian', 'Oren', 'Steve']
```

## 5. python加载

```python
import yaml
import argparse

# 1.加载
with open('configs/gfl.yaml', 'r') as f:
    x = yaml.load(f.read(), Loader=yaml.FullLoader)
    y = yaml.dump(x, Loader=yaml.FullLoader)
    print(x['model']['strides'])

# yaml.load(file, Loader=yaml.FullLoader)
# yaml.safe_load(file)
# yaml.load(file, Loader=yaml.CLoader)

# 2.将配置文件导入argparser
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--configs', default='', type=str, help='path')
    args = parser.parse_args()
    return args


args = parse_args()
if args.configs is not None:
    with open(args.configs, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
    key = vars(args).keys()
    for k in default_arg.keys():
        if k not in key:
            print(f"Wrong Arg: {k}")
            assert k in key
    args.set_defaults(**default_arg)
args = parser.parse_args()
```


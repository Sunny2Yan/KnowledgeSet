# Git
git 四个重要对象：Blob(Binary large object)对象，Tree对象，Commit对象，Tag对象。

## 1. git 设置
git 设置默认保存在自己账号下的`.gitconfig`文件中，可以文件中设置

```bash
# 使用 git 时，需要先设置用户名和用户邮箱
git config --global user.name "xxx"
git config --global user.email xxx
git config --list

# 对于特定项目可以设置不同的作者，在对项目操作时会使用特定的用户名，用户邮箱，离开项目后恢复默认的 global 设置
git config --local user.name "xxx"
git config --local user.email "xxx"

# 指定编辑器，默认为 Vim
git config --global core.editor Vscode

# 对命令取别名，也可以到`~/.gitconfig`中修改
git config --global alias.br branch
git config --global alias.st status
git config --global alias.l "log --oneline --graph"

# 设置和取消代理
git config --global http.proxy 127.0.0.1:7890
git config --global https.proxy 127.0.0.1:7890

git config --global --unset http.proxy
git config --global --unset https.proxy
```

## 2. git 使用

### 2.1克隆现有仓库

```bash
# 1.http协议 git clone [url]
git clone https://github.com/xxx

# 2.git协议 git clone user@server:path/xx.git
git clone git@github.com:Sunny2Yan/mmdetection.git

# 3.克隆分支 git clone -b branch_name [url]/git
git clone -b branch_name https://github.com/xxx
```

### 2.2 更新到仓库`.git`

新目录需要 `git init` 初始化，实现 git 对目录的版本控制。该命令会创建一个 `.git` 隐藏目录。若不想被 git 控制，删除 `.git` 目录即可。
`Working Directory --add-->  Staging Area  --commit-->  .git directory(repository)`

```bash
# 1.添加至缓存区
git add xxx    # 将xxx文件安置到缓存区(staging area)，稍后与其他文件一起存储到存储库中
git add *.md   # 将所有md后缀的文件全部添加
git add --all  # 将所有文件全部添加

# 2.检查状态(untracked)
git status
git status -s  # 状态简览(??: 未被跟踪文件前面有，A: 新添加到暂存区，M: 已修改过)

# 3.提交至仓库(.git) 让缓存区的内容永久保存，即保存一个版本
git commit -m "init commit"        # 说明中英文都可以，但不能没有
git commit --allow-empty -m "空的"  # 说明为空的写法，没有实际意义
git commit -a -m "init commit"  # 同时完成git add(全部文件)和git commit

# 4.忽略文件(.gitignore)
*.[oa]   # 以.o / .a结尾的文件
.idea/*  # .idea下面的全部文件
git add -f file_name  # 可以无视上面的忽略规则
git add -fX  # 清除那些被忽略的文件

# 5.查看修改内容
git diff           # 查看尚未暂存的文件更新了哪些部分
git diff --staged  # 查看已暂存的将要添加到下次提交里的内容
```

- `git add` 后改动文件（状态变为Untracked files），需要再次 `git add` 替换缓存区中的原文件；

- `Changes not staged for commit` 说明已跟踪文件的内容发生了变化，但没有放到暂存区，需要 `git add`。

### 2.3 删除/变更文件

```bash
# 1.从git中移除文件(已跟踪)
git rm xxx.md         # 若删除之前修改过且已经放到暂存区域的话，则使用-f
git rm --cached name  # 从仓库移除，但保留本地(如没有.ignore的文件)

# 2.移动文件(重命名)
git mv file_from file_to  # <==>mv README.md README / git rm README.md / git add README
```

- 变更文件名：git 根据文件内容来计算(SHA-1)识别码，所以文件名不重要。

### 2.4 查看记录

```bash
git log  # 查看git记录，从新到旧显示
    -p: 每次提交的差异
    -2: 近两次提交
    --stat: 列出所有被修改过的文件
    --pretty: 指定使用不同于默认格式的方式展示提交历史(oneline, short, full, fuller...)
git log file_name # 查看指定文件记录
git log --oneline --graph  # 获得精简日志
git log -p file_name  # 查看某文件每次commit的内容

# 按用户名查找
git log --oneline --author="xxx"
git log --oneline --author="xxx\|yyy"  #\转义符，|或
# 按关键字查找
git log --oneline --grep="key"
# 在所有commit文件中查找
git log -S "xxx"
# 按时间段查找(since,until固定搭配，表示每天的时间区间，after表示2021-01之后)
git log --oneline --since="9am" --until="12am" --after="2021-01"

# git blame查看代码作者
git blame file_name
git blame -L 5,10 file_name  # 查看该文件5-10行的作者

git reflog  # 查看引用日志(近几个月HEAD和分支引用所指向的历史)
```

### 2.5 撤销操作

```bash
# 1.使用--amend参数改动最后一次的commit
git commit -m "xxx"  # 第一次提交发现漏掉些文件
git add file.md
git commit --amend -m "xxx"  # 第二次提交将代替第一次提交的结果
git commit --amend --no-edit  # 提交新的文件到最后一次commit，本次没有commit

# 2.取消暂存的文件
git reset HEAD xxx.md

# 3.恢复删除的文件（未提交）：（缓存区文件覆盖本地，危险！）
git checkout file_name  # 恢复删除的指定文件
git checkout .          # 恢复删除的全部文件

# 4.回退之前版本（提交后的也可以回退）
git log --oneline  # 查看版本号
git reset 版本号    # 回退版本(--mixed)

# 5.恢复-hard删除的commit
git reset 版本号    # 删除了commit
git reflog         # 查看记录
git reset 版本号    # 恢复commit记录
```

### 2.6 远程仓库

```bash
# 1.查看远程仓库
git remote 
    -v: 查看url, 如 git remote -v
    show: 查看更多信息, 如 git remote show origin
    
# 2.添加远程仓库(用shortname代替url)
git remote add <shortname> <url>
git remote add origin git@github.com:xxx/learning.git

# 3.将远程文件拉回本地(第一次clone，后续pull)
git fetch origin  # 可以在本地访问远程的分支（本地有了，但不属于自己）
git merge         # 将fetch的分支merge到本地分支
git pull = git fecth + git merge  # 拉取后合并到当前分支

# 4.推送到远程仓库
git push <remote_name> <branch_name>  # 先 git pull 关联(git pull origin main --allow-unrelated-histories)
git push origin dev

# 5.远程仓库的移除与重命名(不是分支！)
git remote rename origin name_1
git remote rm name_1
```

### 2.7 打标签

```bash
# 1.查看标签
git tag
git tag -l "v1.1*"  # 查看版本1.1相关的标签

# 2.创建标签(lightweight轻量级标签、annotated附注标签)
git tag v1.3  # 创建轻量级标签（类似于不会改变的分支对象）
git tag -a v1.4 -m "my version 1.4"  # 创建附注标签（包含完整对象）
git tag -a v1.5 版本号  # 对过去的提交打标签

# 3.推送至共享服务器(git push不会推送标签至远程仓库服务器上)
git push origin [tagname]  # 推送一个标签
git push origin --tags    # 推送全部标签

# 4.检出标签(?)
# 因为tag不能像分支一样来回移动，所以Git中并不能真的检出一个标签，若要工作目录与仓库中特定的标签版本完全一样：
git checkout -b version2 v2.0.0
```

## 3. 分支使用

分支是 `./git/refs/heads/` 中的一个 40 字节的文件。
分支类似于一个指针，另有一个 HEAD 指针指向分支指针，表示当前分支。

```bash
# 1.新建分支
git branch xxx
git branch -m aaa bbb  # 更改分支名称

# 2.删除分支
git branch -d xxx
git branch -D xxx  # 强制删除没有合并的分支

# 3.切换分支
git checkout xxx  # 移动HEAD指针
git checkout -b xxx  # 切换到不存在的分支（创建并切换）

# 4.合并分支
git checkout master
git merge xxx  # 将xxx合并到master（需要先解决冲突）
git merge --abort # 发生冲突时，简单地退出合并，恢复到运行合并前的状态

git diff
git diff --ours  # 查看合并引入了什么
git diff --their  # 查看合并的结果与别人的有什么不同
git diff --base  # 来查看文件在两边是如何改动的
git checkout --ours / --theirs  # bao'li

# 5.分支管理
git branch -v  # 查看每个分支的最后一次提交
git branch --merged  # 查看哪些分支已经合并到当前分支
git branch --no-merged  # 查看哪些分支没有合并到当前分支

# 6.分支开发流
master: 分支上保留完全稳定的代码(保存已经发布或即将发布的代码)
develop: 后续开发或者测试稳定性(可以被合并入master的分支)
特性分支: 短期开发模块

# 7.远程分支
git clone 分支后，本地有一个master分支指向origin/master，当远程有人提交时，远程的master向前移动，但本地的origin/master不会移动，需要 git fetch origin.
git push origin --delete branch_name  # 删除远程分支

# 8.变基(将提交到某一分支上的所有修改都移至另一分支上，即改变基底)
         c4 <--test   test                      test
          |         |                        |
c0 <-- c1 <-- c2 <-- c3 <-- c4' ==>  c0 <-- c1 <-- c2 <-- c3 <-- c4'
               |                            |
              master                         master
git checkout test
git rebase master
git checkout master
git merge test  # <==> git merge test 但分支更整洁，没有分叉
```

## 4. 服务器上的 Git

建立与合作者都有权力访问、且可从那里推送和拉取资料的公用仓库。

### 4.1 协议
Git 可以使用四种主要的协议来传输资料：本地协议（Local），HTTP 协议，SSH（Secure Shell）协议及 Git 协议。

- **本地协议（Local protocol）**：远程版本库就是硬盘内的某一个目录。即，团队中每一个成员都对一个共享的文件系统（如挂载的 NFS）拥有访问权，或者多人共用同一台电脑的情（可能发生灾难性的损失）；

    ```bash
    git clone /opt/git/project.git  # clone/push/pull...
    git remote add local_proj /opt/git/project.git  # 添加本地库到git项目
    ```

- **HTTP 协议**：使用 HTTP 协议的用户名／密码的基础授权，免去设置 SSH 公钥；

    ```bash
    git clone https://example.com/gitproject.git
    ```

- **SSH 协议**：用 SSH 协议作为传输协议；

    ```bash
    git clone ssh://user@server/project.git
    git clone user@server:project.git  # scp式的写法，等同于上面
    ```

- **Git 协议**：监听在一个特定的端口（9418），类似于 SSH服务，但是访问无需任何授权。要让版本库支持 Git 协议，需要先创建一个 git-daemon-export-ok 文件，除此之外没有任何安全措施。要么谁都可以克隆这个版本库，要么谁也不能。这意味这，通常不能通过 Git 协议推送。由于没有授权机制，一旦你开放推送操作，意味着网络上知道这个项目 URL 的人都可以向项目推送数据。

### 4.2 服务器上搭建 Git

1. 把现有仓库导出为裸仓库（不包含当前工作目录的仓库）；

    ```bash
    git clone --bare project project.git  # project.git/... 不同于git init
    git init project  # project/.git/...
    ```

2. 把裸仓库放到服务器上；
    假设存在一个 `git.example.com` 的服务器，并支持 SSH 连接，现想把 Git 仓库放到 `/opt/git` 目录下。

    ```bash
    scp -r project.git user@git.example.com:/opt/git  # 上传上去
    git clone user@git.example.com:/opt/git/project.git  # 之后就可以从服务器clone
    
    # 用户添加可写权限
    ssh user@git.example.com
    $ cd /opt/git/my_project.git
    $ git init --bare --shared
    ```

托管平台：GitWeb（基于网页的简易查看器）、GitLab（功能较全）、第三方托管等。


```bash
# 3.添加子模块
git submodule add third-party/fedgems-python-client  https://gitlab.enncloud.cn/FL-Engine/fedgems-python-client
```



```bash
# github 项目结构
# -build 构建脚本
# -dist 编译出来的发布版
-docker 
-bin  # 脚本文件
-docs  # 文档
-examples  # 示例文件
-scripts  
-src  # 源码
-test  # 测试脚本
  #-.babelrc Babel 交叉编译的配置
  #-.eslintrc ESLint 测试配置
  -.gitignore # 哪些文件不要上传到 GitHub
  #-.gitattributes 文件属性，如 EOL、是否为二进制等
  -LICENSE  # 授权协议
  -setup.cfg
  -setup.py  # 使用setupools进行项目打包
  -README.md  # 自述文件，里面至少得写：1.项目的特色; 2.各种 Badge 比如 CI 的; 3.怎么下载安装; 4.示例运行/使用; 5.怎么编译（尤其 C/C++，要把编译器、要装什么工具、依赖库全部写清楚。要带版本号！）; 6.怎么参与
```




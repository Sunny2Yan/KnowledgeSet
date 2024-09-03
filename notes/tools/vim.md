# VIM

vim 共三种模式，命令模式（Command mode），输入模式（Insert mode）和底线命令模式（Last line mode）。

- Command mode：初始状态，输入为命令。
- Insert mode：编辑文本。
- Last line mode：执行退出、保存等命令。（q, w, qa!...）

Command -(iao)-> Insert；Insert -(esc)-> Command；Command -(:)-> Last line；Last line -(enter)-> Command

## 1. Command mode
按 `esc` 进入命令模式

```bash
gg  # 光标回到行首
ggdG  # 删除全部文本（必须先回到行首，且区分大小写）
ggvG  # 全选
ggyG  # 全选并复制

yy  # 单行复制
dd  # 单行删除
p  # 粘贴

# 搜索词
/word | ?word  # 光标之（下 / 上）查找某个词
n | N  # （下一个 / 上一个）查找的词

# 改写文件
i | a    # 在光标的（当前 / 下一个）位置开始输入文本
u  # 撤销本次进入文件后的所有操作
ctrl + r  # 撤销 u 的操作，即还原撤销
```

## 2.Insert mode

## 3.Last line mode
按 `:` 进入最后一行模式

```bash
# 1. 进入或退出文件
:w | :w! | :q | :q! | :wq | :wq!  # 保存与退出；

# 2. 文件查看
:set nu | :set nonu  # 显示（不显示）行号；
:n1,n2s/word1/word2/g  # n1 与 n2 行之间寻找 word1 字符串，并将该字符串取代为 word2
:%s/word1/word2/g  # 从第一行到最后一行寻找 word1 字符串，并将该字符串取代为 word2 
:%s/word1/word2/gc  # 添加用户确认步骤

# 3. 
:w filename        # 当前内容写入另一个文件；
:r filename        # 将另一个文件的内容写入到当前文件；
:n1 n2 w filename  # n1到n2行的内容复制到另一个文件；

:! command  # 暂时离开vim 到Command模式下（eg: :!ls）
```

## vim配置文件

在用户主目录下建立个.vimrc文件并配置，root账户在/root/下建立一个.vimrc文件。

```
".vimrc
"This is xxxx's vimrc

set nocompatible "不使用兼容模式
set nu "显示行号
syntax on "语法高亮
set ruler "显示标尺
set showcmd "出入的命令显示出来，看的清楚些
set scrolloff=3 "光标移动到buffer的顶部和底部时保持3行距离
set laststatus=2 "总是显示状态行
set noeb "去掉输入错误的提示声音
set autoindent "自动缩进

set expandtab "(是否在缩进和遇到 Tab 键时使用空格替代;使用 noexpandtab 取消设置)
set tabstop=4 "用多少个空格来显示一个制表符，只是用来显示。
set softtabstop=4 "(软制表符宽度,设置为非零数值后使用 Tab 键和 Backspace 时光标移动的格数等于该数值,但实际插入的字符仍受 tabstop 和 expandtab 控制);
"一般不要使用该设置 或 使该值等于你想要的一个制表符的宽度
set shiftwidth=4 "指用>>或<<进行缩进的空格数,例如set shiftwidth=20,再按>>就向左移动20个空格的距离.
"这20个空格的距离是用tabstop来转换的,例如tabstop=5,那按>>的结果就是用4个tab来填充.

set hls "高亮显示搜索结果 等同于 set hlsearch
set showmatch "高亮显示匹配的括号
set whichwrap+=<,>,h,l "允许backspace和光标键跨越行边界
"set cursorline "在光标当先行底部显示一条线，以标识出当前光标所在行
"set mouse=a "鼠标总是可用
set showcmd

set smartindent "暂时还不清楚做什么用的

"set encoding=utf-8 "这个目前还不确定需不需要配置，该怎么配置，这是VIM内部使用的编码方式
"set fileencoding=utf-8 "不需要配置
"Vim 启动时会按照它所列出的字符编码方式逐一探测即将打开的文件的字符编码方式，并且将 fileencoding 设置为最终探测到的字符编码方式
set fileencodings=ucs-bom,utf-8,cp936 "这里很重要，这一行表示vim编码格式依次选择
"解释：cp936是为了支持GBK编码的

set ignorecase "搜索时忽略大小写
set cindent "使用C样式的缩进
autocmd FileType make set noexpandtab "当文件类型是make的时候，set noexpandtab

set statusline=%F%m%r%h%w\ [%{&ff}\|%Y]\ [%04l,%04v\|%p%%*%L]   "vim状态栏的显示信息


" File: vimrc.txt
" Author: deep_thoughts
" Date: 14.10.2021
" Last Modified Date: 14.10.2021
" Last Modified By: deep_thoughts

set nocompatible " be iMproved, required
set noic
syntax on
filetype off " required
set encoding=utf-8
set termencoding=utf-8
set fileencodings=utf-8,gbk,latin1
set nu
set cursorcolumn
set cursorline
set hlsearch
set tabstop=4
set shiftwidth=4
set expandtab
set autoindent
set cul
set cuc
call pathogen#infect()
set t_Co=256
set background=dark
execute pathogen#infect()
set statusline+=%#warningmsg#

" set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*
let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0

filetype plugin indent on
let g:livepreview_previewer = 'open -a Preview'  " latex实时预览
let g:pymode_options_max_line_length = 120  " 设置一行的最大长度
let g:pymode_lint_options_pep8 = {'max_line_length': g:pymode_options_max_line_length}
let g:pymode_options_colorcolumn = 1  "xian'shi

call plug#begin('~/.vim/plugged')
Plug 'xuhdev/vim-latex-live-preview', { 'for': 'tex' }
Plug 'scrooloose/nerdtree'
Plug 'Xuyuanp/nerdtree-git-plugin'
Plug 'sheerun/vim-polyglot'
Plug 'Vimjas/vim-python-pep8-indent'
Plug 'Chiel92/vim-autoformat'
Plug 'scrooloose/nerdcommenter'

" Plug 'zxqfl/tabnine-vim'
" Plug 'davidhalter/jedi-vim'
call plug#end()
" Plug 'python-mode/python-mode', { 'branch': 'develop' }
let g:header_field_author = 'deep_thoughts'
let g:header_field_author_email = ''
map <F4> :AddHeader<CR>
highlight Cursor guifg=white guibg=black
highlight iCursor guifg=white guibg=steelblue
set guicursor=n-v-c:block-Cursor
set guicursor+=i:ver100-iCursor
set guicursor+=n-v-c:blinkon0
set guicursor+=i:blinkwait10

" Nerd Commenter配置
" Add spaces after comment delimiters by default
let g:NERDSpaceDelims = 1
" Use compact syntax for prettified multi-line comments
let g:NERDCompactSexyComs = 1
" Align line-wise comment delimiters flush left instead of following code indentation
let g:NERDDefaultAlign = 'left'
" Set a language to use its alternate delimiters by default
let g:NERDAltDelims_java = 1
" Add your own custom formats or override the defaults
let g:NERDCustomDelimiters = { 'c': { 'left': '/**','right': '*/' } }
" Allow commenting and inverting empty lines (useful when commenting a region)
let g:NERDCommentEmptyLines = 1
" Enable trimming of trailing whitespace when uncommenting
let g:NERDTrimTrailingWhitespace = 1
" Enable NERDCommenterToggle to check all selected lines is commented or not
" let g:NERDToggleCheckAllLines = 1
let mapleader=","
set timeout timeoutlen=3000
highlight Cursor guifg=white guibg=black
highlight iCursor guifg=white guibg=steelblue
set guicursor=n-v-c:block-Cursor
set guicursor+=i:ver100-iCursor
set guicursor+=n-v-c:blinkon0
set guicursor+=i:blinkwait10

" let python excution in VIM
imap <F5> <Esc>:w<CR>:!clear;python %<CR>
```

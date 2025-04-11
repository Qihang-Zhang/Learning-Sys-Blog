---
categories:
  - Profiling
  - In progress
date: 2025-03-02
draft: true
links:
  - index.md
readtime: 1
slug: sglang profiling
authors:
  - <qihang>
---
# SGLang profiling
In this post, I am going to introduce common profiling tools and methods. Specifically, I will introduce the tools and methods I used to profile SGLang and report the results on 8\*A100 and 8\*H100.
<!-- more -->
## Overview
[TOC]

## Shell set-up
This is the recommended shell set-up by [zhaochenyang20](https://github.com/zhaochenyang20) in [Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/engineer/uv/readme.md).
<!-- fold -->
<details>
<summary>
  Click to expand
</summary>


```bash
## Git 相关

# 创建新的 branch
alias gcb="git checkout -b"

# 提交 commit
alias gcm="git commit --no-verify -m"

# 切换 branch
alias gc="git checkout"

# 推送本地新创建的 branch 到远端
alias gpso='git push --set-upstream origin "$(git symbolic-ref --short HEAD)"'

# 推送本地 branch 到远端
alias gp="git push"

# 查看本地 branch
alias gb="git branch"

# 拉取远端 branch
alias gpl="git pull --no-ff --no-edit"

# 添加所有文件
alias ga="git add -A"

# 设置远端 branch
alias gbst="git branch --set-upstream-to=origin/"

# 查看 commit 树
alias glg="git log --graph --oneline --decorate --abbrev-commit --all"

# 查看 commit 表格
alias gl="git log"

## python 相关

# 运行 python
alias py="python"

# 运行 uv pip install，快到飞起
alias upip="python -m uv pip install"

# 运行 ipython
alias ipy="ipython --TerminalInteractiveShell.shortcuts '{\"command\":\"IPython:auto_suggest.resume_hinting\", \"new_keys\": []}'"

## Devlop 需求

# 运行 pre-commit
alias pre="pre-commit run --show-diff-on-failure --color=always --all-files"


# 安装当前目录下的包
alias pi="python -m uv pip install .[all]"

# 查看当前路径下的文件
alias le="less"

# 查看历史命令
alias his="history"

# 查看当前路径下的文件树
alias tr="tree -FLCN 2"

# 查看当前路径下的文件夹
alias trd="tree -FLCNd 2"

# 每 0.1 秒读取当前路径下的磁盘空间使用量，特别在下载模型的时候很好用
alias wd="watch -n 0.1 du -hs"

# 流式读取文件的尾部内容，可以在 tmux 中提交任务然后退出 tmux，在命令行使用这个命令查看任务的 log
alias tf="tail -f"


# 设置文件权限为完全透明
alias c7="chmod 777 -R"

# 打开当前路径
alias op="open ."

# 用 cursor 打开某个文件，用 vscode 同理
alias cur="cursor"

# 用 vscode 打开某个文件
alias cod="code"

# 打开配置文件
alias ope="cursor /data/qihang/.zshrc"

# 用 cursor 打开当前路径下的文件
alias co="cursor ."

# 快速查看 GPU 使用情况
alias nvi="nvidia-smi"

# 每 1 秒查看 GPU 使用情况，需要先 pip install gpustat
alias gpu="watch -n 1 gpustat"

# 创建 tmux 会话
alias tns="tmux new -s"

# 列出 tmux 会话
alias tls="tmux ls"

# 重新登录回到某个 tmux 会话
alias tat="tmux attach -t"

# 重新加载 zsh 配置
alias sz="source /data/qihang/.zshrc"

# 重新加载 bash 配置
alias zb="source /data/qihang/.bashrc"

# 杀死进程
alias k9="kill -9"

# 格式化代码
alias bp="black *.py && black *.ipynb"

# 杀死自己名下的所有 python 进程，慎用
alias kp="ps aux | grep '[p]ython' | awk '{print \$2}' | xargs -r kill -9"

# 删除 ipynb 文件的输出
alias nbs='find . -name "*.ipynb" -exec nbstripout {} \;'

# 用 soft方式重置 git 提交
alias grs="git reset --soft"

## 服务器管理

# 用人类可理解的格式查看当前路径下磁盘空间使用量
alias duh="du -hs"

# 直观查看当前目录下哪些文件或文件夹占用空间最多
alias dus="du -sh --exclude=proc --exclude=sys --exclude=dev * .[^.]* | sort -hr"

# 删除超过 1 天未被访问的 tmp 文件
alias ftt="find /tmp -type f -atime +1 -delete"

# 删除所有 7 天前创建的临时文件和目录
alias ftm="find /tmp -time +7 -exec rm -rf {} +"

# 查看磁盘空间使用量

alias dfh="df -h"

# 设置 huggingface 的 token

export HF_TOKEN="************************"

# 设置 huggingface 的 cache 路径，请一定配置好，避免一个集群重复下载某个模型多次

export HF_DATASETS_CACHE="/data/.cache/huggingface/datasets"
export HF_HOME="/data/.cache/huggingface"

# 设置个人默认路径，我一般连带着所有数据一起放在 /data 下
export HOME="/data/qihang"

# 设置 ray 的 cache 路径，如果不用 ray 不太需要管
export RAY_ROOT_DIR="/data/.cache/ray"

# 设置 wandb 的 api key
export WANDB_API_KEY="*******************8"

# 设置 LD_LIBRARY_PATH，这是配置 flash attention 踩的坑，遇到问题可以参考
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# 用于分配显卡的函数，al k 可以分配 k 张空闲的卡

function al() {
    local num_gpus=$1
    local mem_threshold=1000  # 阈值为 1000MB
    
    echo "Looking for $num_gpus free GPUs..."
    echo "Checking GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits
    
    # 先收集所有符合条件的 GPU，存入数组
    local gpu_ids=($(nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits | 
                   awk -v threshold=$mem_threshold -F, '{
                       gsub(/ /, "", $1); 
                       gsub(/ /, "", $2);
                       if ($1 + 0 < threshold) print $2
                   }'))
    
    local found_count=${#gpu_ids[@]}
    
    if [ $found_count -ge $num_gpus ]; then
        # 只取需要的数量
        local selected_gpus=(${gpu_ids[@]:0:$num_gpus})
        local gpu_string=$(IFS=,; echo "${selected_gpus[*]}")
        export CUDA_VISIBLE_DEVICES=$gpu_string
        echo "✅ CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
        return 0
    else
        echo -e "\033[1;31mError: Requested $num_gpus GPUs but only found $found_count free GPUs\033[0m"
        echo -e "\033[1;31m⚠️  GPU 分配失败！请求 $num_gpus 个 GPU，但只找到 $found_count 个空闲 GPU\033[0m"
        unset CUDA_VISIBLE_DEVICES
        return 1
    fi
}

# 硬性分配显卡

cu() {
    # 将输入的数字字符串转换为逗号分隔的格式
    local devices=$(echo $1 | sed 's/./&,/g' | sed 's/,$//')
    export CUDA_VISIBLE_DEVICES="$devices"
    echo "已设置 CUDA_VISIBLE_DEVICES=$devices"
}

## 激活虚拟环境

function ca() {
    env_name="$1"
    # 根据环境名构造激活脚本的路径
    activation_script="$HOME/.python/${env_name}/bin/activate"

    # 检查激活脚本是否存在
    if [ ! -f "$activation_script" ]; then
        echo "Error: 激活脚本 '$activation_script' 不存在"
        return 1
    fi

    # 激活虚拟环境
    # 注意：使用 source 激活后，当前 shell 环境会被修改
    source "$activation_script"

    ceiling="===== Activated Env: ${env_name} ====="
    echo "$ceiling"

    # 输出当前 python 路径和版本
    python_path=$(which python)
    echo "Python 路径：$python_path"
    python --version

    # 如果你想检查环境是否切换成功，可以考虑检查环境变量
    # 比如，在虚拟环境中通常会设置 VIRTUAL_ENV 变量
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "===== NO!! Environment switch failed ====="
        return 1
    else
        echo "===== YES!! Environment switched to: $VIRTUAL_ENV ====="
    fi
}


# 读取时间戳，用于给 log 做标记

function now() {
    date '+%Y-%m-%d-%H-%M'
}

# 我个人的工作路径

alias sgl="cd /data/qihang/sglang/python"
alias rlhf="cd /data/qihang/OpenRLHF-SGLang/openrlhf"
alias vllm="cd /data/qihang/vllm/"
alias docs="cd /data/qihang/sglang/docs"
alias test="cd /data/qihang/sglang/test"
alias awe="cd /data/qihang/Awesome-ML-SYS-Tutorial"

# uv 相关

# 查看当前虚拟环境
alias uvv="uv venv"

# 创建虚拟环境

# python3 -m venv ~/.python/sglang

# 激活虚拟环境

# source ~/.python/sglang/bin/activate

## 分配 1 张 GPU
al 1

## 激活 sglang 环境
ca sglang
sleep 1
clear
```
</details>

## Docker
### container
#### docker container run

- Description: Create and run a new container from an image 
- Usage: `docker container run [OPTIONS] IMAGE [COMMAND] [ARG...]` 
- Aliases: `docker run` 
- Example:

```bash
docker container run --gpus all --name sglang-container lmsysorg/sglang:latest
```

frequently used options:

- `--name`: specify the name of the container
- `--gpus all`: allow the container to use all GPUs
- `-it`: allow the container to interact with the terminal
- `--shm-size`: specify the size of the shared memory
- `-v`: mount a volume from the host to the container
    - can be added multiple times
    - format: `-v <host_path>:<container_path>`
        - e.g. `-v ~/.cache/huggingface:/root/.cache/huggingface`
- `-d`: run the container in the background

#### docker container exec

- Description: Execute a command in a running container
- Usage: `docker container exec [OPTIONS] CONTAINER COMMAND [ARG...]`
- Aliases: `docker exec`
- Example:

```bash
docker container exec -it sglang-container bash
```

```bash
docker exec -it sglang-container sh -c "echo 'Hello, World'"
```

## Profiling
### draft

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max reqeuest concurrency:                not set   
Successful requests:                     10        
Benchmark duration (s):                  7.90      
Total input tokens:                      1960      
Total generated tokens:                  1000      
Total generated tokens (retokenized):    1000      
Request throughput (req/s):              1.27      
Input token throughput (tok/s):          248.25    
Output token throughput (tok/s):         126.66    
Total token throughput (tok/s):          374.91    
Concurrency:                             3.38      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   2666.99   
Median E2E Latency (ms):                 2666.62   
---------------Time to First Token----------------
Mean TTFT (ms):                          1344.00   
Median TTFT (ms):                        1335.83   
P99 TTFT (ms):                           1377.56   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           13.36     
Median ITL (ms):                         12.98     
P95 ITL (ms):                            13.45     
P99 ITL (ms):                            16.14     
Max ITL (ms):                            128.64    
==================================================











***References:***

- [如何配置一台爽快的开发机器](https://zhuanlan.zhihu.com/p/23440683394)
  
- [sglang doc](https://docs.sglang.ai/references/benchmark_and_profiling.html)
  
- [sglang benchmark](https://github.com/sgl-project/sglang/tree/main/benchmark)

- [How to use docker for sglang](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/engineer/how-to-use-docker)


\bibliography



```bash
# 建立 SSH 本地端口转发
ssh -N -f -L localhost:8888:localhost:8888 xzha593@foscsmlprd01

# 配置 HTTP/HTTPS 代理
export http_proxy=http://squid.auckland.ac.nz:3128
export HTTP_PROXY=http://squid.auckland.ac.nz:3128
export https_proxy=http://squid.auckland.ac.nz:3128
export HTTPS_PROXY=http://squid.auckland.ac.nz:3128

# 下载文件
scp -r xzha593@foscsmlprd01:/data/xzha593/GitHub/my-solution-active-learning-2.2/results_FBYG15K_0.2_run5.log .

# 上传文件
scp -r ./my-solution-active-learning-2.2 xzha593@foscsmlprd01:/data/xzha593/GitHub

# 通过 SSH 连接服务器
ssh xzha593@foscsmlprd01

# 查看显卡占用情况
fuser -v /dev/nvidia2

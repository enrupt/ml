alias ll='ls -alF'

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook"
export SPARK_HOME=/usr/local/opt/spark-3.5.1-bin-hadoop3-scala2.13
export PATH=$SPARK_HOME/bin:$PATH
export HADOOP_HOME=/usr/local/opt/hadoop/libexec/
export PATH=$HADOOP_HOME/bin:$PATH
export SPARK_MASTER="spark://localhost:9000"
export SPARK_MASTER_HOST=localhost
export SPARK_MASTER_PORT=7077

export PYSPARK_PYTHON=/opt/anaconda3/bin/python
#export PYSPARK_PYTHON=jupyter
export HADOOP_CONF_DIR="$HADOOP_HOME/etc/hadoop"
export HDFS_NAMENODE_USER="a1"
export HDFS_DATANODE_USER="a1"
export HDFS_SECONDARYNAMENODE_USER="a1"
export YARN_RESOURCEMANAGER_USER="a1"
export YARN_NODEMANAGER_USER="a1"
export PATH=$PATH:/opt/anaconda3/bin/jupyter




#THIS MUST BE AT THE END OF THE FILE FOR SDKMAN TO WORK!!!
export SDKMAN_DIR="$HOME/.sdkman"
[[ -s "$HOME/.sdkman/bin/sdkman-init.sh" ]] && source "$HOME/.sdkman/bin/sdkman-init.sh"

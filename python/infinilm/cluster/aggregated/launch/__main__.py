from .launcher import main

# 支持 `python -m infinilm.cluster.aggregated.launch` 启动，同时保留入口脚本。
if __name__ == "__main__":
    main()

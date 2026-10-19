from .launcher import main

# 支持 `python -m infinilm.cluster.launch` 启动，同时保留 console script 入口。
if __name__ == "__main__":
    main()

import ray
import time

# 启动一个单机集群，假装有 4 个 CPU 核心
ray.init(num_cpus=4)


@ray.remote
def worker_task(i):
    time.sleep(1)
    # 获取当前运行这个任务的机器 IP（单机都是同一个 IP）
    return f"Task {i} finished on {ray.util.get_node_ip_address()}"


# 同时发起 4 个任务
futures = [worker_task.remote(i) for i in range(4)]
print(ray.get(futures))

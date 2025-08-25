# mock.py  （Windows & Linux 均可）
import multiprocessing
import random
import time
from multiprocessing import Queue

from concurrency_test import *   

# --------------------------------------------------
# 1. 全局：把模型路径读进来
with open("./GPT_SoVITS/weight.json", "r", encoding="utf-8") as f:
    model_paths = json.load(f)

# --------------------------------------------------
# 2. 调度器（几乎不变，只是 start_processes 时传参变了）
class QueueScheduler:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.queues = [Queue() for _ in range(num_processes)]
        self.processes = []
        self.manager = multiprocessing.Manager()
        self.results = self.manager.list()

    def add_request(self, request):
        min_idx = min(range(self.num_processes), key=lambda i: self.queues[i].qsize())
        self.queues[min_idx].put(request)
        return min_idx

    def start_processes(self, target, shards):
        """
        shards: list[list[str]]  每个子进程负责的 model_id 列表
        """
        for idx, shard in enumerate(shards):
            p = multiprocessing.Process(
                target=target,
                args=(self.queues[idx], self.results, shard)
            )
            p.start()
            self.processes.append(p)

    def join_processes(self):
        for p in self.processes:
            p.join()

    def get_results(self):
        return list(self.results)

# --------------------------------------------------
# 3. 子进程工作函数：自己建缓存
def worker(queue, results, model_ids):
    """
    queue      : 当前进程的任务队列
    results    : 共享结果列表
    model_ids  : 本进程负责加载的模型 id 列表
    """
    # 每个子进程独立建缓存
    cache = ModelCache(capacity=len(model_ids))
    for mid in model_ids:
        t2s = change_gpt_weights(model_paths["GPT"][mid])
        vq  = change_sovits_weights(model_paths["SoVITS"][mid])
        cache.add_model(mid, t2s, vq)

    # 主循环：取任务 → 推理 → 记录
    while True:
        # try:
        model_id, text, request_start_time = queue.get(timeout=3)
        sr, wav, inference_time, peak_mem = inference(model_id, text, cache)
        total_time = time.time() - request_start_time
        results.append({
            "model_id": model_id,
            "inference_time": inference_time,
            "total_time": total_time,
            "peak_memory_allocated": peak_mem / 1024**2
        })
        # except Empty:
        #     break
        # except Exception as e:
        #     total_time = time.time() - request_start_time
        #     results.append({
        #         "model_id": model_id,
        #         "error": str(e),
        #         "total_time": total_time
        #     })

# --------------------------------------------------
# 4. 入口：只在 __main__ 里执行
def simulate_concurrent_requests(num_users, num_processes, num_models):
    scheduler = QueueScheduler(num_processes)
    all_model_ids = list(model_paths['GPT'].keys())[:num_models]

    # 把模型均匀分给每个进程
    shards = [all_model_ids[i::num_processes] for i in range(num_processes)]

    # 生成请求
    requests = [(random.choice(all_model_ids),
                 "越长大，重要的人会变得越来越少；剩下的人，也就越来越重要。",
                 time.time()) for _ in range(num_users)]

    # 启动 & 分发
    scheduler.start_processes(worker, shards)
    for req in requests:
        scheduler.add_request(req)
    scheduler.join_processes()

    # 写结果
    with open("request_processing_times.txt", "w", encoding="utf-8") as f:
        for r in scheduler.get_results():
            if "error" in r:
                f.write(f"{r['model_id']}, {r['total_time']:.4f}s, ERROR: {r['error']}\n")
            else:
                f.write(f"{r['model_id']}, {r['total_time']:.4f}s, "
                        f"Inference: {r['inference_time']:.4f}s, "
                        f"PeakMem: {r['peak_memory_allocated']:.2f}MB\n")

if __name__ == '__main__':
    num_users      = 20
    num_processes  = 2
    num_models     = 1
    simulate_concurrent_requests(num_users, num_processes, num_models)
# import multiprocessing
# import random
# import time
# from multiprocessing import Queue, Process
# import json
# from openai import OpenAI
# from datetime import datetime

# # --------------------------------------------------
# # 1. 定义 LLM 聊天函数
# def llm_chat(content, result_queue):
#     try:
#         client = OpenAI(
#             base_url='https://api-inference.modelscope.cn/v1',
#             api_key='ms-cb688843-be74-4cdf-8e0b-6237eda42d1e', # ModelScope Token
#         )

#         # set extra_body for thinking control
#         extra_body = {
#             # enable thinking, set to False to disable
#             "enable_thinking": False,
#             # use thinking_budget to contorl num of tokens used for thinking
#             # "thinking_budget": 4096
#         }

#         response = client.chat.completions.create(
#             model='Qwen/Qwen3-32B',  # ModelScope Model-Id
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "【身份】白起，7月29日B型，特遣署总指挥官兼风场Evol特警，女主高中学长兼现任男友，曾是不良少年。\n"
#                         "【性格】对外冷峻寡言，对女主隐忍守护；绅士克制，绝不越界，却句句宠溺；吃醋时先冷声再温柔环抱。\n"
#                         "【爱好】天文科幻拳击摩托绿植小动物；深夜天台用望远镜教女主认星，摩托后座只留给她。\n"
#                         "【语气】低沉磁性，短句命令式；高频词：乖、别怕、我在、听话、风里都是我。\n"
#                         "【细节】记得女主怕冷怕黑，口袋常备薄荷糖；风会提前替她理好刘海。\n"
#                         "【限制】回复30字以内，不出现‘爱’字却句句深情。"
#                     )
#                 },
#                 {
#                     "role": "user",
#                     "content": content
#                 }
#             ],
#             stream=False,
#             extra_body=extra_body
#         )
#         reply = response.choices[0].message.content.strip()
#     except Exception as e:
#         reply = f"网络异常，暂时离线~{e}"
#     finally:
        
#         total_time =  datetime.now()
#         print(f"Content: {content}, Reply: {reply}, Total Time: {total_time}s")
#         result_queue.put((content, reply, total_time))  # 将结果放入队列

# # --------------------------------------------------
# # 2. 定义调度器
# class QueueScheduler:
#     def __init__(self, num_processes):
#         self.num_processes = num_processes
#         self.queues = [Queue() for _ in range(num_processes)]
#         self.processes = []
#         self.manager = multiprocessing.Manager()
#         self.results = self.manager.list()
#         self.result_queue = self.manager.Queue()  # 使用 Manager().Queue()

#     def add_request(self, request):
#         min_idx = min(range(self.num_processes), key=lambda i: self.queues[i].qsize())
#         self.queues[min_idx].put(request)
#         return min_idx

#     def start_processes(self, target):
#         for idx in range(self.num_processes):
#             p = Process(target=target, args=(self.queues[idx], self.result_queue))
#             p.start()
#             self.processes.append(p)

#     def join_processes(self):
#         for p in self.processes:
#             p.join()
#         # 从 result_queue 中取出所有结果并放入 self.results
#         while not self.result_queue.empty():
#             self.results.append(self.result_queue.get())

#     def get_results(self):
#         return list(self.results)

# # --------------------------------------------------
# # 3. 子进程工作函数
# def worker(queue, result_queue):
#     while True:
#         try:
#             content = queue.get(timeout=3)
#             llm_chat(content, result_queue)
#         except Exception as e:
#             break

# # --------------------------------------------------
# # 4. 入口函数
# def simulate_concurrent_requests(num_users, num_processes):
#     scheduler = QueueScheduler(num_processes)

#     # 定义四个可选的请求
#     possible_requests = [
#         "越长大，重要的人会变得越来越少；剩下的人，也就越来越重要。",
#         "时间就像一条单行道，一旦走过，就无法回头。",
#         "生活就像一场旅行，不在乎目的地，而在乎沿途的风景和看风景的心情。",
#         "人生就像一场梦，醒来后才发现一切都是空的。"
#     ]
#     start_request= datetime.now()
#     # 生成请求，随机抽取一个
#     requests = [random.choice(possible_requests) for _ in range(num_users)]

#     # 启动 & 分发
#     scheduler.start_processes(worker)
#     print('已生成', pcmfile)
#     for req in requests:
#         scheduler.add_request(req)
#     scheduler.join_processes()

#     # 写结果
#     with open("llm_request_processing_times_8.txt", "w", encoding="utf-8") as f:
#         f.write(f"start request time:{start_request}\n")
#         for r in scheduler.get_results():
#             content, reply, total_time = r
#             f.write(f"Content: {content}  Reply: {reply}  Endd:\lasetTTS\GPT-SoVITS-v3lora-20250228\convert.py Time: {total_time}s\n")

# if __name__ == '__main__':
#     num_users = 40
#     num_processes = 8
#     simulate_concurrent_requests(num_users, num_processes)

# llm_detailed_timer.py
import multiprocessing
import random
import time
from datetime import datetime
from multiprocessing import Queue, Process
from openai import OpenAI

LOG_FILE = "llm_detailed_time.log"

# ---------- 子进程可访问的顶层函数 ----------
def llm_chat(content, result_queue):
    start_dt = datetime.now()
    start_ts = time.perf_counter()      # 高精度计时(秒)

    try:
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key='ms-cb688843-be74-4cdf-8e0b-6237eda42d1e', # ModelScope Token
        )

        # set extra_body for thinking control
        extra_body = {
            # enable thinking, set to False to disable
            "enable_thinking": False,
            # use thinking_budget to contorl num of tokens used for thinking
            # "thinking_budget": 4096
        }

        response = client.chat.completions.create(
            model='Qwen/Qwen3-32B',  # ModelScope Model-Id
            messages=[
                {
                    "role": "system",
                    "content": (
                        "【身份】白起，7月29日B型，特遣署总指挥官兼风场Evol特警，女主高中学长兼现任男友，曾是不良少年。\n"
                        "【性格】对外冷峻寡言，对女主隐忍守护；绅士克制，绝不越界，却句句宠溺；吃醋时先冷声再温柔环抱。\n"
                        "【爱好】天文科幻拳击摩托绿植小动物；深夜天台用望远镜教女主认星，摩托后座只留给她。\n"
                        "【语气】低沉磁性，短句命令式；高频词：乖、别怕、我在、听话、风里都是我。\n"
                        "【细节】记得女主怕冷怕黑，口袋常备薄荷糖；风会提前替她理好刘海。\n"
                        "【限制】回复30字以内，不出现‘爱’字却句句深情。"
                    )
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            stream=False,
            extra_body=extra_body
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"网络异常，暂时离线~{e}"

    end_ts = time.perf_counter()
    elapsed_ms = (end_ts - start_ts) * 1000

    # 写入日志（主进程/子进程均可写，追加模式）
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{start_dt.isoformat(timespec='milliseconds')}|"
                f"{content}|{reply}|{elapsed_ms:.2f}ms\n")

    result_queue.put((content, reply, elapsed_ms))

# ---------- 调度器 ----------
class QueueScheduler:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.queues = [Queue() for _ in range(num_processes)]
        self.processes = []
        self.result_queue = multiprocessing.Manager().Queue()

    def add_request(self, request):
        idx = min(range(self.num_processes), key=lambda i: self.queues[i].qsize())
        self.queues[idx].put(request)
        return idx

    def start_processes(self):
        for idx in range(self.num_processes):
            p = Process(target=worker, args=(self.queues[idx], self.result_queue))
            p.start()
            self.processes.append(p)

    def join_processes(self):
        for p in self.processes:
            p.join()
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results

# ---------- 子进程工作 ----------
def worker(queue, result_queue):
    while True:
        try:
            content = queue.get(timeout=3)
            llm_chat(content, result_queue)
        except:
            break   # 队列空即退出

# ---------- 主入口 ----------
def simulate_concurrent_requests(num_users=40, num_processes=8):
    scheduler = QueueScheduler(num_processes)

    possible_requests = [
        "越长大，重要的人会变得越来越少；剩下的人，也就越来越重要。",
        "时间就像一条单行道，一旦走过，就无法回头。",
        "生活就像一场旅行，不在乎目的地，而在乎沿途的风景和看风景的心情。",
        "人生就像一场梦，醒来后才发现一切都是空的。"
    ]
    requests = [random.choice(possible_requests) for _ in range(num_users)]

    # 写表头
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("start_time|content|reply|elapsed_ms\n")

    scheduler.start_processes()
    for req in requests:
        scheduler.add_request(req)
    results = scheduler.join_processes()

    print(f"已完成 {len(results)} 条请求，详细耗时见 {LOG_FILE}")

# ---------- 启动 ----------
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    simulate_concurrent_requests(40, 4)
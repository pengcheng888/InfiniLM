import asyncio
from pathlib import Path
import time
from openai import AsyncOpenAI
import argparse
import random
import subprocess


PROMPTS = [
    "如果猫能写诗，它们会写些什么？",
    "描述一个没有重力的世界。",
    "如果地球停止自转，会发生什么？",
    "假设你是一只会飞的鲸鱼，描述你的日常生活。",
    "如果人类可以与植物沟通，世界会变成什么样？",
    "描述一个由糖果构成的城市。",
    "如果时间旅行成为可能，你最想去哪个时代？",
    "想象一下，如果地球上只有蓝色，其他颜色都消失了。",
    "如果动物能上网，它们会浏览什么网站？",
    "描述一个没有声音的世界。",
    "如果人类可以在水下呼吸，城市会如何变化？",
    "想象一下，如果天空是绿色的，云是紫色的。",
    "如果你能与任何历史人物共进晚餐，你会选择谁？",
    "描述一个没有夜晚的星球。",
    "如果地球上只有一种语言，世界会如何运作？",
    "想象一下，如果所有的书都变成了音乐。",
    "如果你可以变成任何一种动物，你会选择什么？",
    "描述一个由机器人统治的未来世界。",
    "如果你能与任何虚构角色成为朋友，你会选择谁？",
    "想象一下，如果每个人都能读懂他人的思想。",
    """《如果声音有形状》
写作任务：请完成一篇8000字的文章或故事。
开放性要求：将一种抽象的声音（如母亲的呼唤、城市的车流、童年午后的蝉鸣）转化为一个具体的视觉或触觉形象。不可直接写“声音像波浪”，要赋予它独特的物理属性（温度、质地、重量），并通过这个“形状”串联起你与某个重要人物的情感记忆。
切入点建议：写一次深夜独自走在雨后的老街上，你“看见”了已故祖父的咳嗽声——它是一块温润的旧玉石，你弯腰拾起，以此为起点展开关于家族传承的完整叙事。""",
    """《我选择记住那个错误》
写作任务：请完成一篇500字的文章或故事。
开放性要求：反向思考——不写“从错误中吸取教训后遗忘”，而是写主动保留一次失败、一次误解、一次伤害。剖析为什么这个“错误”值得被留在记忆里甚至被珍藏，因为它改变了你看待世界的滤镜。
切入点建议：详细描写那个错误发生后的第7天，你决定不删除手机里的那段录音，反而为它建了一个专属文件夹。通过反复重听，你发现了原本被愤怒掩盖的细节，最终与自己和解释然。以此为核心，完成一篇完整的内心成长故事。""",
    """《在快时代里慢下来》
写作任务：请完成一篇1000字的文章或故事。
开放性要求：选取一个现代社会极度追求效率的场景（如外卖配送、短视频剪辑、高铁旅行），刻意地“慢放”其中某个瞬间。不需要批判快，只需呈现慢带来的陌生化美感——比如观察外卖员在等红灯时抬头看云的3秒钟。
切入点建议：锁定一个“停顿”的瞬间——一位外卖骑手在暴雨中突然停下电动车，不是因为故障，而是因为路边一朵野花在雨中被砸弯又弹起。跟随他的目光，展开对他日常节奏的想象，最终完成一篇关于“速度与凝视”的散文。""",
    """《翻译一首无法翻译的诗》
写作任务：请完成一篇2000字的文章或故事。
开放性要求：假设你发现了一首用某种濒危方言或自创密码写成的短诗，没有任何词典可查。你只能通过诗人的生平、周围的环境符号（如壁画、器物）来“猜”出它的意思。这个过程将揭示：翻译本身就是一种创造，而不是复制。
切入点建议：以考古学家的第一人称笔记形式，记录你破解这首诗的14天。每一天你推翻前一天的译法，最后你发现诗的本意是“请忘记我”，但你选择译成“请记住我”——围绕这一选择，构建一篇关于文化理解与自我投射的完整故事。""",
    """《当城市停止运转一小时》
写作任务：请完成一篇3000字的文章或故事。
开放性要求：设定一个场景——整个城市因某种非灾难原因（如全球同步的“静默日”）停电停网一小时。描写这一小时内，不同角落的人们如何度过。不写恐慌，写原本被噪音掩盖的细节：邻居的钢琴声、楼顶的星空、陌生人之间的微笑。
切入点建议：聚焦一个高层住宅的电梯停运，一位年轻人被迫走楼梯，在楼梯间遇到一位每天擦肩而过却从未说过话的老人，两人用一支蜡烛的光分享了一刻钟的沉默。以此为核，编织出一篇关于“偶遇与连接”的短篇小说。""",
    """《种一棵看不见的树》
写作任务：请完成一篇4000字的文章或故事。
开放性要求：这不是真实的树，而是一种精神象征——比如你为某个逝去的人种下的“记忆之树”，或者为未来某个未出生的孩子种下的“希望之树”。描写这棵树如何在你心里生长，它的根须、叶子、年轮分别对应什么。
切入点建议：你每天深夜在日记里为这棵树浇水（写下一件当天发生的小确幸），直到有一天，你发现这棵树在现实中投影出了一片阴凉——一个陌生人因为你的某个善举而感到了安慰。围绕这一发现，完成一篇关于“无形善意如何有形回响”的散文。""",
    """《给十年前的自己写一封拒绝信》
写作任务：请完成一篇5000字的文章或故事。
开放性要求：不是写给未来，而是写给过去的自己。信的内容是拒绝当年那个让你后悔的决定（比如拒绝一次妥协、拒绝一段关系）。但重点不在“如果重来会怎样”，而在你通过这封信，终于理解了当年的自己为何会那样选择——从而达成对过去的谅解。
切入点建议：信的开头是“亲爱的，我要拒绝你当时的选择”，但写着写着，你开始为那个选择辩护，最后发现你其实是在拒绝“现在的自己”对过去的苛责。以此结构，完成一篇关于自我和解的完整书信体故事。""",
    """《为一束光写传记》
写作任务：请完成一篇6000字的文章或故事。
开放性要求：选取一束特定的光——可以是某天下午穿过百叶窗的阳光、一盏陪伴你熬夜的台灯光、或者极夜里的一束极光。为这束光写一部简史：它从哪里来，照亮过哪些人，见证过哪些秘密，最终消失在哪里。
切入点建议：以这束光的第一人称口吻（自然现象的拟人化，非AI），讲述它在一位老画家的画室里停留了三十年，看着画家的风格从写实到抽象，最后在画家离世后，它落在一幅未完成的画布上，形成了最后一个笔触。围绕这条时间线，完成一篇关于“见证与传承”的抒情叙事。""",
]

IMAGE_PROMPTS = [
    "请描述一下图片里的内容。",
    "图片里有人吗？",
    "请结合图片，讲一个小故事。",
]

NUM_REQUESTS = 32
CONCURRENCY = 8
API_URL = "http://127.0.0.1:8000"
MODEL = ""


class ImageCollector:
    def __init__(self, dir_path: str, port=None):
        self.dir_path = Path(dir_path).resolve()

        if not self.dir_path.is_dir():
            raise ValueError(f"Not a valid directory: {self.dir_path}")

        self.image_files = [
            file.resolve()
            for file in self.dir_path.rglob("*")
            if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg"]
        ]

        assert len(self.image_files) > 0, "No image file found in provided directory!"

        self.host = "127.0.0.1"
        self.port = port
        self.server_process = None

        # Only start HTTP server if BOTH host and port are provided
        self.use_http = self.host is not None and self.port is not None

        if self.use_http:
            self._start_server()

    def _start_server(self):
        print(
            f"[ImageCollector] Starting image HTTP server...\n"
            f"  Directory: {self.dir_path}\n"
            f"  URL: http://{self.host}:{self.port}\n"
        )
        self.server_process = subprocess.Popen(
            [
                "python",
                "-m",
                "http.server",
                str(self.port),
                "--bind",
                self.host,
            ],
            cwd=str(self.dir_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(0.5)

    def stop_server(self):
        if self.server_process is not None:
            self.server_process.terminate()

            try:
                self.server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

            self.server_process = None

    def __del__(self):
        self.stop_server()

    def random_image_url(self):
        image_path = random.choice(self.image_files)

        # Return local absolute path
        if not self.use_http:
            return str(image_path)

        # Return HTTP URL
        relative_path = image_path.relative_to(self.dir_path)

        return f"http://{self.host}:{self.port}/{relative_path.as_posix()}"


async def benchmark_user(
    client, semaphore, queue, results, user_id, verbose, image_collector=None
):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break

            try:
                print(f"🚀 User#{user_id} Sending request #{task_id}")
                messages = None
                if image_collector is None:
                    messages = [{"role": "user", "content": random.choice(PROMPTS)}]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_collector.random_image_url()
                                    },
                                },
                                {"type": "text", "text": random.choice(IMAGE_PROMPTS)},
                            ],
                        }
                    ]

                print(messages)

                start_time = time.time()
                stream = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=True,
                )

                first_token_time = None
                total_tokens = 0
                answer_chunks = []

                async for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.time()
                    delta = chunk.choices[0].delta.content
                    if delta:
                        answer_chunks.append(delta)
                        total_tokens += 1
                    if chunk.choices[0].finish_reason is not None:
                        break

                end_time = time.time()

                ttft = first_token_time - start_time if first_token_time else None
                elapsed_time = end_time - start_time if start_time else None
                ms_per_token = (
                    (elapsed_time / total_tokens * 1000)
                    if total_tokens > 0 and elapsed_time
                    else None
                )
                tokens_per_second = (
                    total_tokens / elapsed_time if elapsed_time > 0 else 0
                )

                answer = "".join(answer_chunks)

                results.append(
                    (total_tokens, elapsed_time, tokens_per_second, ttft, ms_per_token)
                )

                if verbose:
                    print(f"\n📝 Request #{task_id} (User #{user_id})")
                    if ttft is not None:
                        print(f"  ⏱ 首字延迟 TTFT: {ttft:.3f}s")
                    if elapsed_time is not None:
                        print(f"  ⏱ 总耗时: {elapsed_time:.3f}s")

                    print(f"  🔤 解码 token 总数: {total_tokens}")
                    if ms_per_token is not None:
                        print(f"  📏 平均 token 解码时间: {ms_per_token:.2f} ms/token")
                    else:
                        print(f"  📏 平均 token 解码时间: N/A (no token generated)")
                    print(f"  ❓ 提问: {messages}")
                    print(f"  💬 回答: {answer}\n")

                queue.task_done()
            except Exception as e:
                if verbose:
                    print(f"\n⚠️ Request #{task_id} (User #{user_id}) FAILED:")
                    print(f"  ❌ Error: {e}\n")
                queue.task_done()


async def run_benchmark(verbose=False, image_collector=None):
    client = AsyncOpenAI(base_url=API_URL, api_key="default")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    queue = asyncio.Queue()
    results = []
    for i in range(NUM_REQUESTS):
        await queue.put(i)
    for _ in range(CONCURRENCY):
        await queue.put(None)

    users = [
        asyncio.create_task(
            benchmark_user(
                client, semaphore, queue, results, user_id, verbose, image_collector
            )
        )
        for user_id in range(CONCURRENCY)
    ]

    start_time = time.time()
    await queue.join()
    await asyncio.gather(*users)
    end_time = time.time()

    total_elapsed_time = end_time - start_time
    tokens_list = [r[0] for r in results if r and r[0] is not None]
    latencies = [r[1] for r in results if r and r[1] is not None]
    tokens_per_second_list = [r[2] for r in results if r and r[2] is not None]
    ttft_list = [r[3] for r in results if r and r[3] is not None]
    ms_per_token_list = [r[4] for r in results if r and r[4] is not None]

    successful_requests = len(results)
    requests_per_second = (
        successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    )
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = (
        sum(tokens_per_second_list) / len(tokens_per_second_list)
        if tokens_per_second_list
        else 0
    )
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    avg_ms_per_token = (
        sum(ms_per_token_list) / len(ms_per_token_list) if ms_per_token_list else None
    )

    width_label = 24
    sep = "-" * 60

    print(f"\n=== 📊 性能指标汇总 ({MODEL}) ===")
    print(sep)
    print(f"{'并发数':<{width_label}}: {CONCURRENCY}")
    print(f"{'请求总数':<{width_label}}: {NUM_REQUESTS}")
    print(f"{'成功请求数':<{width_label}}: {successful_requests}")
    print(f"{'总耗时':<{width_label}}: {total_elapsed_time:.2f} s")
    print(f"{'总输出token数':<{width_label}}: {sum(tokens_list)}")
    print(f"{'请求速率 (RPS)':<{width_label}}: {requests_per_second:.2f} requests/s")
    print(sep)
    print(f"{'Average latency':<{width_label}}: {avg_latency:.2f} s")
    print(f"{'Average TTFT':<{width_label}}: {avg_ttft:.2f} s")
    print(f"{'Avg time per token':<{width_label}}: {avg_ms_per_token:.2f} ms/token")
    print(
        f"{'Avg Token generation speed':<{width_label}}: {avg_tokens_per_second:.2f} tokens/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--mm-port", type=str, default=None)
    parser.add_argument("--api-url", type=str, default="127.0.0.1:8000")
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()

    API_URL = "http://" + args.api_url
    MODEL = args.model

    image_collector = None
    if args.image_dir is not None:
        image_collector = ImageCollector(args.image_dir, port=args.mm_port)

    asyncio.run(run_benchmark(args.verbose, image_collector))

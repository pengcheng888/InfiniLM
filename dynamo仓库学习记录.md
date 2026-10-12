# 查看 Dynamo 仓库

> 先查看/workspace_codex/dynamo/仓库。

<details><summary>2 previous messages</summary>

> 我先做只读检查：查看仓库状态、顶层结构、当前分支和关键说明文件，先建立代码库全貌，不做任何修改。
>
> 初步确认当前在 `main`，工作区干净，项目版本是 `1.6.0`。这是一个 Rust 核心、Python 扩展层和 Kubernetes 部署层并存的大型单仓库；我再看各主要目录的入口、构建脚本和近期提交，整理成可操作的仓库导览。
</details>

已查看 `/workspace_codex/dynamo` 仓库，目前：

- 分支：`main`，与 `origin/main` 对齐
- 工作区：干净，无未提交修改
- 当前版本：`1.6.0`
- 技术栈：Rust 核心、Python 扩展层、Go Kubernetes Operator
- 核心目录：
  - [`lib/`](./lib)：runtime、LLM、KV Router、KVBM、sidecar 等 Rust crate
  - [`components/src/dynamo/`](./components/src/dynamo)：frontend、planner、router 和各推理后端
  - [`deploy/`](./deploy)：Operator、Helm、Inference Gateway、可观测性
  - [`tests/`](./tests)：Python、后端、路由、运行时和部署测试
  - [`recipes/`](./recipes)：模型部署配方
  - [`docs/fern/`](./docs/fern)：Fern 文档站
- 仓库各关键子目录还有独立的 `AGENTS.md`，后续修改时需要按目标文件所在作用域读取。
- 本次仅做了只读检查，没有修改文件或运行测试。

> 将/workspace_codex/dynamo/examples/README.md翻译成中文，保存为新的.md文件。

我会使用 `dynamo-docs` 技能，因为目标位于 `examples/`，该目录受仓库文档规范约束。我先读取该技能、适用的 `AGENTS.md` 和原文，再新建中文版本，保留原文件不变。

> 将/workspace_codex/dynamo/examples/README.md翻译成中文，保存为新的.md文件。

<details><summary>8 previous messages</summary>

> 继续处理；上一轮只完成了规范与原文读取，没有写入文件。我会新建 `examples/README.zh-CN.md`，完整翻译正文并保持命令、路径和链接目标不变。
>
> 现在开始写入中文文件。翻译会沿用原文的章节结构、代码块和全部链接，只将说明文字、标题与表格文本本地化；新文件按仓库规范使用 Markdown SPDX 注释头。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./examples/README.zh-CN.md` (+105 -0)
>
> </details>
>
> 中文文件已创建。接下来我会核对差异、Markdown 结构和链接目标，并运行文档检查器；若检查器只能扫描整个 `examples/` 树，我会区分新文件问题与仓库已有告警。
>
> 结构核对发现新文件已通过基础差异检查；另外原英文文件本身有两个相对链接指向当前仓库中不存在的路径。为保证这是忠实翻译，我暂时保留了这些原始链接，正在用文档检查器确认新文件没有额外引入问题，并逐项比对中英文链接集合。
>
> 英文源文件在处理期间发生了更新：当前版本已移除 TokenSpeed 和 Nemotron Speech 两项，而中文草稿基于先前读取的版本仍包含它们。这也正是检查器报出的两个断链；我会将中文文件同步到当前英文源文件后重新验证。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./examples/README.zh-CN.md` (+0 -5)
>
> </details>
>
> 中文文件现已与当前英文版本同步，两个失效条目已移除。最后验证将检查文档规范、空白错误、链接集合一致性，以及原文在验证过程中是否再次变化。
</details>

已完成翻译并保存为 [examples/README.zh-CN.md](./examples/README.zh-CN.md)。

原英文文件未修改。验证结果：

- 文档检查通过，0 个错误
- 中英文均包含 9 个标题、26 个链接
- `git diff --check` 通过

> /workspace_codex/dynamo/examples/custom_backend/hello_world/README.md翻译成中文，保存为新的.md文件。

<details><summary>4 previous messages</summary>

> 我会继续按 `dynamo-docs` 技能处理，因为这是 `examples/` 下的 README 翻译。先读取当前原文和该目录适用的文档规范，再新建同目录下的 `README.zh-CN.md`，不修改英文文件。
>
> 原文共 114 行，包含架构图、两组本地运行命令、预期输出以及 Kubernetes 部署步骤。我将保留所有命令、环境变量、文件名和代码标识符不变，并把原文的普通 “Note” 引用转换为仓库要求的 `[!NOTE]` 格式。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./examples/custom_backend/hello_world/README.zh-CN.md` (+117 -0)
>
> </details>
>
> 中文文件已写入。现在进行针对性验证：检查 SPDX、链接和 Markdown 结构，并逐块比较中英文代码围栏，确保架构图、命令和预期输出没有因翻译发生变化。
</details>

已完成翻译并保存为 [README.zh-CN.md](./examples/custom_backend/hello_world/README.zh-CN.md)。

英文原文件未修改。文档检查和 `git diff --check` 均通过，代码块、命令及链接与原文一致。

> 在单个 GPU 上运行单个模型，那么单独使用推理引擎通常已经足够。
> 如果在多个GPU上，聚合式，启动多个推理引擎服务。Dynamo可以提供哪些能力呢

如果多个 GPU 上启动了多个相同模型的聚合式推理服务，Dynamo 的价值主要不在单个推理引擎内部，而在多个服务实例之间的统一编排和流量调度。

典型请求路径如下：

```text
客户端
  ↓
Dynamo Frontend（统一 OpenAI 兼容 API）
  ↓
Dynamo Router
  ├─ 推理引擎实例 A：Prefill + Decode
  ├─ 推理引擎实例 B：Prefill + Decode
  └─ 推理引擎实例 C：Prefill + Decode
```

Dynamo 可以提供这些能力：

| 能力 | 作用 |
|---|---|
| 统一服务入口 | 多个推理实例对外暴露为一个 OpenAI 兼容 API，客户端不需要知道具体 worker 地址 |
| 服务发现 | 自动发现新增、重启或退出的推理 worker |
| 负载均衡 | 根据 worker 的运行状态和负载分配请求，避免某个 GPU 过载而其他 GPU 空闲 |
| KV 感知路由 | 根据请求前缀与各 worker KV cache 的重叠程度路由，尽量复用已有 KV cache，减少重复 prefill |
| 多种路由策略 | 支持轮询、KV 感知、最小负载、设备感知以及自定义路由策略 |
| 弹性扩缩容 | Planner 可以根据吞吐量、Time To First Token（TTFT）和 Inter-Token Latency（ITL）等目标调整副本数量 |
| 故障处理 | 进行 worker 健康检查，停止向故障实例发送新请求；部分配置支持请求迁移等容错能力 |
| KV 缓存管理 | 通过 KVBM 管理 GPU、CPU、SSD 或远端存储之间的 KV cache，具体支持程度取决于推理后端 |
| Kubernetes 编排 | 使用 DynamoGraphDeployment 管理模型服务、worker 副本、资源配置和滚动部署 |
| 可观测性 | 汇总请求延迟、吞吐量、路由决策、KV cache 和 worker 状态等指标 |

需要区分两种多 GPU 方式：

1. **一个推理引擎实例使用多个 GPU**

   例如 vLLM 使用 Tensor Parallelism（TP）在 4 个 GPU 上加载一个模型。模型切分、集合通信、连续批处理和 CUDA 执行仍由 vLLM、SGLang 或 TensorRT-LLM 负责。Dynamo 不会替代这些能力。

2. **启动多个推理引擎副本**

   例如 4 个 GPU 各运行一个相同模型的 worker。此时 Dynamo 可以统一入口、发现实例、平衡负载、进行 KV 感知路由，并结合 Planner 扩缩容。这是聚合式部署中 Dynamo 更有价值的场景。

聚合式部署下，每个 worker 都同时执行 prefill 和 decode，因此不会获得“预填充池与解码池独立扩缩容”的收益。若工作负载中长输入、长输出差异明显，或者 prefill 和 decode 对 GPU 资源的需求不平衡，才更适合进一步考虑分离式部署。

> 统一服务入口与服务发现，这个能力对应的介绍文档在哪里？请分析

<details><summary>4 previous messages</summary>

> 我会从本地文档和实现入口两条线查找：先定位“Frontend 统一入口”和“服务发现”的正式文档，再核对它们在代码中的实际边界，避免把 Router、Runtime discovery 和 Kubernetes Service 混为一谈。此次只分析，不修改文件。
>
> 初步结果表明这两个能力分散在不同层次的文档中：统一入口主要由 Frontend 文档说明，服务发现则分为架构层的 discovery plane 和部署层的后端选择。接下来我会精读这些候选页面，确认哪些是主文档、哪些只是配置或操作补充。
>
> 目前可以确认：仓库内没有名为 `service-discovery.md` 的独立页面。根 README 仍链接到线上 `/kubernetes-deployment/advanced-platform/service-discovery`，但本地权威内容已经集中到 Architecture 的 “Discovery Plane” 和 Runtime Configuration。我要再检查导航、重定向与实现注册流程，判断这是有意合并还是文档入口存在漂移。
>
> 从重定向可以确认，独立的 Service Discovery 页面已被删除并明确迁移到 Architecture 的 `#discovery-plane`。这不是缺页；但根 README 仍使用旧线上 URL，依赖重定向才能到达新内容。下面我会把“客户端入口”“模型注册”“端点成员变化”三段实际链路串起来。
</details>

结论：这两个能力没有集中在同一篇文档中。“统一服务入口”归 Frontend 文档，“服务发现”归 Runtime/Architecture 文档。

**推荐阅读顺序**

1. [Frontend Overview](./docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md:8)

   最直接介绍统一入口：Dynamo Frontend 是 API Gateway，提供 OpenAI 兼容 HTTP API、KServe gRPC、请求预处理、路由和响应格式化。第 45 行明确说明后端调用 `register_model` 后会被自动发现。

2. [Architecture](./docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.md:33)

   这是服务发现当前最主要的设计文档：

   - 第 35-44 行：`DistributedRuntime → Namespace → Component → Endpoint` 的服务组织结构
   - 第 54-63 行：Discovery Plane
   - Kubernetes：`DynamoWorkerMetadata` + `EndpointSlice`
   - 本地或裸金属：默认 etcd
   - 开发环境：还支持 `file` 和 `mem`

3. [Runtime Configuration](./docs/fern/pages/reference/components/runtime-configuration.mdx:72)

   这是配置层面的权威参考：

   - `--namespace`：限定哪些组件可以互相发现
   - `--endpoint`：`dyn://namespace.component.endpoint`
   - `--discovery-backend`：`kubernetes`、`etcd`、`file`、`mem`
   - `DYN_DISCOVERY_BACKEND`：对应环境变量

4. [Expose the Frontend](./docs/fern/pages/kubernetes/model-deployment/expose-the-frontend.md:8)

   说明如何把统一入口暴露给外部客户端。Operator 会创建 `<name>-frontend` ClusterIP Service，然后可通过 Ingress、LoadBalancer 或 GAIE 暴露。

5. [Frontend Configuration Reference](./docs/fern/pages/reference/components/frontend-configuration.mdx:641)

   列出 Frontend 对外提供的完整接口，包括 `/v1/chat/completions`、`/v1/models`、`/health`、`/metrics`、`/openapi.json` 和 Swagger UI。

**两个能力如何配合**

```text
客户端
  │
  ▼
稳定外部地址 / <name>-frontend Service
  │
  ▼
Dynamo Frontend
  │
  ├─ 监听模型注册和 worker 成员变化
  ├─ 建立当前可用 worker 集合
  └─ Router 从集合中选择目标
        │
        ▼
  聚合式推理 worker A/B/C
```

它们是两个不同层次：

- **统一服务入口**属于南北向流量：客户端只访问 Frontend，不感知 worker 数量和地址。
- **服务发现**属于内部控制面：Frontend、Router、Planner 和 worker 查找彼此，并监听实例加入、更新和退出。
- Discovery 不直接处理用户请求，也不等同于负载均衡；它向 Router 提供当前可用的目标集合。
- Request Plane 才负责实际发送请求，默认使用 TCP。Discovery backend 和请求传输方式彼此独立。

实现上，worker 通过 `register_model` 注册模型信息；底层构建 `ModelDeploymentCard` 并写入 discovery backend，见 [local_model.rs](./lib/llm/src/local_model.rs:465)。Frontend 初始化对应的 `DistributedRuntime`，见 [main.py](./components/src/dynamo/frontend/main.py:394)。运行时通过 `list_and_watch` 监听端点变化，并动态维护实例集合，见 [client.rs](./lib/runtime/src/component/client.rs:896)。

**文档现状分析**

仓库以前有独立的 `service-discovery.md`，现在已经删除。旧 URL 被明确重定向到 Architecture 的 `#discovery-plane`，见 [docs.yml](./docs/fern/docs.yml:529)。因此 [README.md](./README.md:293) 中的旧线上地址不是当前源文件位置，而是依赖重定向。

当前文档还有一处明显不一致：Frontend Overview 第 35-37 行把 etcd 和 NATS 写成前置条件，但 [README.md](./README.md:260) 和 Architecture 已说明，本地可使用 `file` discovery，Kubernetes 使用原生 discovery，默认 TCP/ZMQ 路径也不要求外部 etcd 或 NATS。因此 Frontend Overview 的前置条件更像旧部署方式残留，不应理解为所有部署的硬性要求。

> /workspace_codex/dynamo/docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md翻译成中文，保存为新的.md文件。

<details><summary>6 previous messages</summary>

> 我会使用 `dynamo-docs` 技能处理这次 Fern 文档翻译。按照仓库的本地化约定，新文件不会放在英文源文件旁边，而会镜像保存到 `docs/fern/translations/zh-CN/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md`；我先核对翻译目录、链接目标和页面规范。
>
> 目标翻译文件目前不存在。关联页面中只有 Router Overview 已有中文版本，其余 Frontend、配置参考和 `nvext` 页面尚未翻译，因此我会让 Router 链接留在中文镜像内，其余链接按规范回指英文基础页面，避免产生断链。
>
> 准备写入中文镜像页面：保留所有命令、YAML、参数名和产品名，翻译 frontmatter、正文、标题与表格。该页面由 Fern 根据相同相对路径自动与英文页配对，因此无需修改 `docs/fern/index.yml`。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./docs/fern/translations/zh-CN/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md` (+101 -0)
>
> </details>
>
> 中文页面已创建。接下来验证 Frontmatter、SPDX、镜像路径、链接解析和代码块一致性；重点检查那些回指英文基础树的七级相对路径是否都能落到真实文件。
>
> 针对新页面的文档检查已通过，0 个错误；报告的 5 条导航警告来自仓库中其他既有页面。代码块数量和内容与英文原文一致，所有中文镜像链接及回退英文链接都已解析到实际文件。我再运行本地化链接解析器的专项测试并复核最终差异。
</details>

已完成翻译，并按 Fern 本地化目录规范保存为：

[中文 Frontend Overview](./docs/fern/translations/zh-CN/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md)

英文原文件未修改。验证结果：

- Docs lint：0 个错误
- `git diff --check`：通过
- 代码块与英文原文一致
- 所有中文及英文回退链接均有效
- 未运行 `fern check`，当前环境未安装 Fern CLI

> 请保存到/workspace_codex/dynamo/docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md同级目录下。

<details><summary>4 previous messages</summary>

> 明白。我会按你的明确位置要求覆盖默认的 Fern 翻译镜像约定：把已生成的中文页移动到英文文件同级目录，并命名为 `overview.zh-CN.md`，不保留镜像目录中的重复副本。由于仍是文档目录变更，我会按 `dynamo-docs` 规范重新核对后执行。
>
> 移动后文件的相对位置发生变化，因此原先用于翻译镜像的七级回退链接必须恢复为与英文页相同的同级相对链接。我现在同时完成文件移动和链接调整，避免新位置出现断链。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./docs/fern/translations/zh-CN/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md` (+6 -6)
>
> </details>
>
> 文件已经移动到指定同级目录。由于 `overview.zh-CN.md` 不属于 Fern 的原生翻译镜像路径，它不会自动出现在语言切换器中；我不会为它添加重复导航项，只验证其作为独立 Markdown 文件的结构和链接。
</details>

已按要求移动到英文文件同级目录：

[overview.zh-CN.md](./docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.zh-CN.md)

原翻译镜像文件已移除，相关链接已按新位置调整。Docs lint 和 `git diff --check` 均通过。

注意：该位置不属于 Fern 的标准翻译镜像目录，因此不会自动接入网站语言切换器，但可作为独立 Markdown 文件使用。

> /workspace_codex/dynamo/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.md翻译成中文，保存为新的.md文件。

<details><summary>4 previous messages</summary>

> 我会继续使用 `dynamo-docs` 技能，并沿用你上一次明确的保存方式：在英文文件同级目录新建 `architecture.zh-CN.md`。正文、标题和表格会翻译，Mermaid 图代码、标识符、环境变量与协议名保持不变。
>
> 原文共 226 行，后半部分是 Mermaid 架构图。现在创建 `architecture.zh-CN.md`：翻译 frontmatter、章节、流程步骤、表格和说明文字；Mermaid 代码整体保持原样，以避免改变图结构或渲染行为。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.zh-CN.md` (+229 -0)
>
> </details>
>
> 中文架构页已创建。现在验证 Markdown/frontmatter、Mermaid 代码逐字一致性、章节数量及空白格式；由于这是按你要求放在同级目录的独立文件，检查时会关闭导航覆盖检查，避免把它误判为新的 Fern 导航页面。
</details>

已完成翻译并保存为同级新文件：

[architecture.zh-CN.md](./docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.zh-CN.md)

英文原文件未修改。Docs lint 和 `git diff --check` 均通过；15 个章节标题与 Mermaid 图代码均和原文保持一致。

该同级文件不会自动接入 Fern 的语言切换器。

> 要求：模型使用/data-aisoft/mechdancer/models/Qwen3-0.6B。GPU0-2 上启动了3个相同模型的聚合式推理服务。
> 分析上述要求，提供启动的命令，以及测试验证步骤。

<details><summary>5 previous messages</summary>

> 这是本地单机、3 个 GPU、3 个聚合式副本的启动方案。我会使用 `dynamo-router-starter` 技能核对当前仓库支持的 Frontend/worker 参数和冒烟测试流程；后端未指定，我先按 vLLM 方案分析，同时检查本地模型目录与可能冲突的端口配置。
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我会先核对仓库中的 vLLM 多实例测试、文件式服务发现参数和路由健康检查脚本，确保命令与当前版本一致。重点会把“一个逻辑模型、三个单卡副本”与“一个模型跨三卡张量并行”明确区分。
>
> 仓库的当前约定已经确认：单机可用 `file` 服务发现，vLLM 在该模式下需要关闭 KV 事件；三个 worker 还必须使用不同的 `DYN_SYSTEM_PORT`。下面继续核对官方聚合式启动脚本和可观测端点，避免把 worker 指标端口误当成推理 API 端口。
>
> 配置形态已经明确：采用 `round-robin` 作为最容易验证的聚合式基线，每个 vLLM worker 同时承担 prefill 和 decode，分别只看到 GPU 0、1、2；Frontend 暴露唯一的 OpenAI API。当前执行环境本身看不到 CUDA 驱动，因此我只能做参数与仓库实现校验，不能在这里实际拉起 GPU worker。
</details>

这里的“聚合式”应理解为：3 个独立的 vLLM 副本，每个副本同时完成 prefill 和 decode，各占用一张 GPU，由一个 Dynamo Frontend 提供统一入口并进行轮询路由。它不是一个模型通过 TP=3 跨三张 GPU 运行。

以下按 vLLM 后端配置。

**启动前检查**

```bash
cd /workspace_codex/dynamo

test -f /data-aisoft/mechdancer/models/Qwen3-0.6B/config.json
nvidia-smi -L
ss -ltnp | rg ':(8000|8081|8082|8083)\b'
```

预期能看到 GPU 0、1、2，并且上述端口没有被占用。

**启动服务**

在一个终端中执行。该终端需要保持运行，按 `Ctrl+C` 会停止所有进程。

```bash
cd /workspace_codex/dynamo

MODEL=/data-aisoft/mechdancer/models/Qwen3-0.6B
SERVED_MODEL=Qwen3-0.6B
NAMESPACE=qwen3-agg

RUN_DIR=$(mktemp -d /tmp/dynamo-qwen3-agg.XXXXXX)
export DYN_FILE_KV="$RUN_DIR/discovery"
mkdir -p "$DYN_FILE_KV"

pids=()
trap 'kill "${pids[@]}" 2>/dev/null; wait' EXIT INT TERM

# 统一入口：只有该进程对客户端暴露推理 API
python3 -m dynamo.frontend \
  --namespace "$NAMESPACE" \
  --discovery-backend file \
  --request-plane tcp \
  --router-mode round-robin \
  --router-min-initial-workers 3 \
  --http-host 0.0.0.0 \
  --http-port 8000 \
  >"$RUN_DIR/frontend.log" 2>&1 &
pids+=("$!")

# 三个聚合式 worker：每个 worker 独占一张 GPU
for gpu in 0 1 2; do
  system_port=$((8081 + gpu))

  CUDA_VISIBLE_DEVICES="$gpu" \
  DYN_SYSTEM_PORT="$system_port" \
  python3 -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL" \
    --namespace "$NAMESPACE" \
    --discovery-backend file \
    --request-plane tcp \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --kv-events-config '{"enable_kv_cache_events":false}' \
    >"$RUN_DIR/worker-gpu${gpu}.log" 2>&1 &

  pids+=("$!")
done

echo "日志目录: $RUN_DIR"
wait -n "${pids[@]}"
```

关键点：

- `CUDA_VISIBLE_DEVICES=0/1/2` 将三个副本分别固定在三张卡上。
- `--tensor-parallel-size 1` 明确每个副本是单卡模型。
- `--served-model-name Qwen3-0.6B` 让客户端不需要使用本地文件路径作为模型名。
- `DYN_SYSTEM_PORT=8081/8082/8083` 必须互不相同。
- `--router-min-initial-workers 3` 防止少于三个副本时过早完成路由器初始化。
- `file` 服务发现适用于同一主机；多节点应改用 Kubernetes 原生发现或 etcd。
- `--enforce-eager` 适合首次启动验证。正式压测时可移除，让 vLLM 使用 CUDA Graph。

**验证步骤**

1. 检查三个 worker 都已就绪：

```bash
for port in 8081 8082 8083; do
  curl --fail --silent "http://127.0.0.1:${port}/health" |
    jq -e '.status == "ready"'
done
```

2. 检查 Frontend 是否发现三个聚合式副本：

```bash
curl --fail --silent \
  http://127.0.0.1:8000/v1/models/Qwen3-0.6B/ready |
  jq
```

重点结果应为：

```json
{
  "ready": true,
  "namespaces": {
    "qwen3-agg": {
      "ready": true,
      "worker_types": {
        "decode": {
          "workers": 3
        }
      }
    }
  }
}
```

可直接做严格判断：

```bash
curl --fail --silent \
  http://127.0.0.1:8000/v1/models/Qwen3-0.6B/ready |
  jq -e '
    .ready == true and
    .namespaces["qwen3-agg"].worker_types.decode.workers == 3
  '
```

聚合式 worker 在 Dynamo 中显示为 `decode` 是正常的，它实际上同时执行 prefill 和 decode。

3. 执行仓库自带的完整冒烟测试：

```bash
python3 \
  .agents/skills/dynamo-router-starter/scripts/check_router_health.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen3-0.6B
```

成功时输出中应包含 `"ok": true`，并且 `/v1/models` 与 `/v1/chat/completions` 都返回 HTTP 200。

4. 连续发送 6 个请求验证轮询路由：

```bash
for i in 1 2 3 4 5 6; do
  curl --fail --silent http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "Qwen3-0.6B",
      "messages": [
        {"role": "user", "content": "只回复 READY"}
      ],
      "temperature": 0,
      "max_tokens": 32
    }' |
    jq -e '.choices | length > 0' >/dev/null ||
    exit 1
done

echo "6 个请求均成功"
```

5. 确认三个 worker 都实际处理过请求：

```bash
curl --silent http://127.0.0.1:8000/metrics |
  rg '^dynamo_frontend_worker_last_time_to_first_token_seconds'
```

正常情况下应看到 3 个不同的 `worker_id`，且各自的 TTFT 值大于 0。也可通过 `nvidia-smi pmon -c 1` 确认 GPU 0、1、2 上均有对应进程。

这些参数依据仓库的 [vLLM 聚合式示例](./examples/backends/vllm/launch/agg.sh:60)和[健康检查说明](./docs/fern/pages/reference/observability/health-checks.mdx:14)。当前执行环境无法访问 CUDA 驱动，因此未在这里实际启动 GPU worker。

> 模型比较小，所以gpu-memory-utilization是0.3就行。按照上述步骤测试功能：3 个独立的 vLLM 副本，每个副本同时完成 prefill 和 decode，各占用一张 GPU，由一个 Dynamo Frontend 提供统一入口并进行轮询路由。

<details><summary>20 previous messages</summary>

> Stream error
>
> Reconnecting... waiting for network
> Connection failed: error sending request
>
> 我会按 `dynamo-router-starter` 技能执行实际启动和冒烟验证，并把显存占用参数改为 `0.3`。先检查 GPU、端口和当前 Dynamo/vLLM 环境，再启动 Frontend 与三个独立 worker；若 CUDA 在受限环境中不可见，我会直接申请在非沙箱环境运行。
>
> <details><summary>Ran 5 commands</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `./.agents/skills/dynamo-router-starter/SKILL.md`
>
> </details>
>
> - Ran `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader`
>
> - Ran `ss -ltnp`
>
> - Ran `pgrep -af 'dynamo\.(frontend|vllm)'`
>
> - Ran `python3 -c 'import dynamo, vllm; print("dynamo", dynamo.__path__); print("vllm", vllm.__version__)'`
>
> - Ran `du -sh /data-aisoft/mechdancer/models/Qwen3-0.6B; test -f /data-aisoft/mechdancer/models/Qwen3-0.6B/config.json; test -f /data-aisoft/mechdancer/models/Qwen3-0.6B/model.safetensors`
>
> </details>
>
> 预检查发现已有一个不属于本次启动的 Frontend 占用 `8000`，我不会终止它，本次改用 `18000`；worker 系统端口改用 `18081–18083`。当前沙箱无法与 NVIDIA 驱动通信，我现在按规则申请在非沙箱环境检查 GPU，确认这是权限隔离还是主机驱动问题。
>
> <details><summary>Ran a command</summary>
>
> - Ran `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader`
>
> </details>
>
> 沙箱外确认 GPU 0–2 均为 A100 80GB 且当前显存空闲，可以进行实际测试。接下来会创建隔离的文件发现目录，使用端口 `18000/18081–18083` 启动服务；三个 worker 的 `--gpu-memory-utilization` 均设为 `0.3`。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `mktemp -d /tmp/dynamo-qwen3-agg.XXXXXX`
> - Ran `mkdir -p /tmp/dynamo-qwen3-agg.HVvmYD/discovery`
> - Ran `env DYN_FILE_KV=/tmp/dynamo-qwen3-agg.HVvmYD/discovery python3 -m dynamo.frontend --namespace qwen3-agg --discovery-backend file --request-plane tcp --router-mode round-robin --router-min-initial-workers 3 --http-host 127.0.0.1 --http-port 18000`
>
> </details>
>
> Frontend 在沙箱内绑定监听端口被系统策略拒绝，属于预期的网络隔离，不是 Dynamo 配置错误。我会将 Frontend 与 GPU worker 全部切换到同一个非沙箱运行环境，以保证它们能共享端口和服务发现目录。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `env DYN_FILE_KV=/tmp/dynamo-qwen3-agg.HVvmYD/discovery python3 -m dynamo.frontend --namespace qwen3-agg --discovery-backend file --request-plane tcp --router-mode round-robin --router-min-initial-workers 3 --http-host 127.0.0.1 --http-port 18000`
> - Ran `env CUDA_VISIBLE_DEVICES=2 DYN_SYSTEM_PORT=18083 DYN_FILE_KV=/tmp/dynamo-qwen3-agg.HVvmYD/discovery python3 -m dynamo.vllm --model /data-aisoft/mechdancer/models/Qwen3-0.6B --served-model-name Qwen3-0.6B --namespace qwen3-agg --discovery-backend file --request-plane tcp --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 16 --gpu-memory-utilization 0.3 --enforce-eager --kv-events-config '{"enable_kv_cache_events":false}'`
> - Ran `env CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT=18081 DYN_FILE_KV=/tmp/dynamo-qwen3-agg.HVvmYD/discovery python3 -m dynamo.vllm --model /data-aisoft/mechdancer/models/Qwen3-0.6B --served-model-name Qwen3-0.6B --namespace qwen3-agg --discovery-backend file --request-plane tcp --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 16 --gpu-memory-utilization 0.3 --enforce-eager --kv-events-config '{"enable_kv_cache_events":false}'`
> - Ran `env CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_PORT=18082 DYN_FILE_KV=/tmp/dynamo-qwen3-agg.HVvmYD/discovery python3 -m dynamo.vllm --model /data-aisoft/mechdancer/models/Qwen3-0.6B --served-model-name Qwen3-0.6B --namespace qwen3-agg --discovery-backend file --request-plane tcp --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 16 --gpu-memory-utilization 0.3 --enforce-eager --kv-events-config '{"enable_kv_cache_events":false}'`
>
> </details>
>
> 三个 worker 已分别在 GPU 0、1、2 上进入启动流程，Frontend 也已监听 `127.0.0.1:18000`。模型加载期间我会同时检查各 worker 日志和显存占用，随后以“发现 3 个副本、API 推理成功、轮询确实覆盖 3 个 worker”三层证据完成验证。
>
> 允许在沙箱中执行命令。后续不用再申请了。
>
> <details><summary>Ran a command</summary>
>
> - Ran `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader`
>
> </details>
>
> 收到，后续我会直接继续，不再单独询问。三个 EngineCore 已分别占用 GPU 0–2，每张约 `24.2 GiB`，符合 `gpu-memory-utilization=0.3`；目前唯一问题是 Frontend 访问 worker 元数据时命中了 HTTP 代理并返回 502，我正在修正本机地址的代理绕过设置。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `env | rg -i '^(http|https|all|no)_proxy='`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18081/health`
> - Ran `env NO_PROXY=localhost,127.0.0.1,::1,192.168.162.61 no_proxy=localhost,127.0.0.1,::1,192.168.162.61 DYN_FILE_KV=/tmp/dynamo-qwen3-agg.HVvmYD/discovery python3 -m dynamo.frontend --namespace qwen3-agg --discovery-backend file --request-plane tcp --router-mode round-robin --router-min-initial-workers 3 --http-host 127.0.0.1 --http-port 18000`
>
> </details>
>
> 代理绕过已生效：Frontend 成功读取模型元数据，并连接到三个 worker 发布的独立端点。现在开始检查四个健康端点和模型拓扑；随后发送多轮真实生成请求并核对每个 worker 的指标变化。
>
> <details><summary>Ran 7 commands</summary>
>
> - Ran `curl --silent --show-error --fail --retry 10 --retry-all-errors --retry-delay 1 http://127.0.0.1:18000/health`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18081/health`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18082/health`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18083/health`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18000/v1/models`
> - Ran `curl --silent --show-error --fail --retry 10 --retry-all-errors --retry-delay 1 http://127.0.0.1:18000/v1/models/Qwen3-0.6B/ready`
> - Ran `python3 .agents/skills/dynamo-router-starter/scripts/check_router_health.py --base-url http://127.0.0.1:18000 --model Qwen3-0.6B --prompt '只回复 READY' --max-tokens 32 --retries 10`
>
> </details>
>
> 基础冒烟测试已经通过：`/v1/models` 和真实 `/v1/chat/completions` 均返回 200。拓扑接口明确报告 `aggregated.workers = 3`；接下来发送 6 个串行请求，并用 Frontend 的按 worker 指标确认轮询覆盖三个不同实例，而不只是假设路由生效。
>
> <details><summary>Ran 10 commands</summary>
>
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' --data '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Return the single word READY."}],"temperature":0,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}'`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' --data '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Return the single word READY."}],"temperature":0,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}'`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' --data '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Return the single word READY."}],"temperature":0,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}'`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' --data '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Return the single word READY."}],"temperature":0,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}'`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' --data '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Return the single word READY."}],"temperature":0,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}'`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' --data '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Return the single word READY."}],"temperature":0,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}'`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18000/metrics`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18081/metrics`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18082/metrics`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18083/metrics`
>
> </details>
>
> 轮询已得到直接证据：6 个串行请求全部返回 `Ready.`，三个独立 worker 的成功请求计数分别为 `2、2、2`；此前冒烟测试额外落在 GPU 1，因此它还有 1 个 `length` 请求。最后我会记录 GPU 索引与显存占用并确认服务仍保持可用，然后交付当前访问地址与复现命令。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader`
> - Ran `curl --silent --show-error --fail http://127.0.0.1:18000/v1/models/Qwen3-0.6B/ready`
>
> </details>
</details>

功能测试通过，服务当前仍在运行。

- 统一入口：`http://127.0.0.1:18000`
- 路由模式：`round-robin`
- 模型：`Qwen3-0.6B`
- 拓扑：3 个独立 aggregated worker，分别绑定 GPU 0、1、2
- `gpu-memory-utilization`：`0.3`
- 每张 GPU 实际占用：约 `24,766 MiB`
- 模型就绪状态：`ready: true`
- Dynamo 发现的副本数：`aggregated.workers = 3`
- 6 个串行请求全部成功，三个 worker 分别处理 `2、2、2` 个请求，轮询路由验证通过
- 仓库自带冒烟测试通过：`"ok": true`

可直接复验：

```bash
curl --fail http://127.0.0.1:18000/health

curl --silent \
  http://127.0.0.1:18000/v1/models/Qwen3-0.6B/ready |
  jq
```

推理测试：

```bash
curl --silent http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Return the single word READY."}
    ],
    "temperature": 0,
    "max_tokens": 8,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }' | jq
```

本次没有使用 `8000`，因为该端口已有一个用户进程运行，未对它进行改动。启动时还发现环境设置了 HTTP 代理；Frontend 访问 worker 的内部地址会返回 502，因此启动 Frontend 时增加了：

```bash
NO_PROXY=localhost,127.0.0.1,::1,192.168.162.61
no_proxy=localhost,127.0.0.1,::1,192.168.162.61
```

本次属于功能冒烟测试，不代表吞吐量或延迟基准测试结果。

> 将启动 命令和测试命令的步骤记录到一个.md文档中。方便我复现。


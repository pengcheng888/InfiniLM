"""Zero-dependency monitoring dashboard served by ClusterFrontend."""

MONITORING_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>InfiniLM Cluster Monitor</title>
<style>
:root {
  --bg:#f3f5f4; --panel:#fff; --text:#17211d; --muted:#65716c;
  --line:#d8dfdb; --accent:#106f64; --accent-bg:#e3f1ee;
  --green:#157347; --green-bg:#e6f4ea; --amber:#8f5b00;
  --amber-bg:#fff1d6; --red:#a4262c; --red-bg:#fde9e9;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:15px 22px}
.header-row{max-width:1480px;margin:auto;display:flex;justify-content:space-between;align-items:center;gap:16px}
.title{display:flex;align-items:center;gap:10px;min-width:0}
h1{font-size:18px;margin:0;letter-spacing:0;white-space:nowrap}
.dot{width:10px;height:10px;border-radius:50%;background:var(--muted)}
.dot.ready{background:var(--green)} .dot.warning{background:var(--amber)}
.header-actions{display:flex;align-items:center;gap:8px;color:var(--muted);white-space:nowrap}
.badge,button{border:1px solid var(--line);background:#fff;border-radius:6px}
.badge{display:inline-flex;align-items:center;padding:5px 8px;font-size:12px;color:var(--accent);background:var(--accent-bg);border-color:transparent}
button{padding:5px 9px;font:inherit;font-size:13px;color:var(--text);cursor:pointer}
button.primary{color:#fff;background:var(--accent);border-color:var(--accent)}
button[aria-pressed="false"]{color:var(--muted)}
a{color:var(--accent);text-decoration:none;border:1px solid var(--line);background:#fff;padding:5px 9px;border-radius:6px;font-size:13px}
main{max-width:1480px;margin:auto;padding:18px 22px 32px}
.error{display:none;margin:0 22px 14px;max-width:1480px;padding:9px 12px;border:1px solid #edc5c7;border-radius:7px;background:var(--red-bg);color:var(--red)}
.summary{display:grid;grid-template-columns:repeat(4,minmax(125px,1fr));gap:10px;margin-bottom:14px}
.metric,.panel{background:var(--panel);border:1px solid var(--line);border-radius:7px}
.metric{padding:12px 13px;min-height:78px}
.label{color:var(--muted);font-size:12px;margin-bottom:7px}
.value{font:600 21px/1 var(--mono);overflow-wrap:anywhere}
.sub{margin-top:6px;color:var(--muted);font-size:12px}
.layout{display:grid;grid-template-columns:minmax(0,2.7fr) minmax(300px,1fr);gap:14px;align-items:start}
.panel-header{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:11px 13px;border-bottom:1px solid var(--line)}
h2{font-size:14px;margin:0}
.wrap{overflow-x:auto}
table{width:100%;min-width:1700px;border-collapse:collapse}
th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;font-size:13px}
th{position:sticky;top:0;background:#fafbfa;color:var(--muted);font-size:12px}
th.num,td.num{text-align:right}
td.mono{font-family:var(--mono);font-size:12px}
.ellipsis{max-width:170px;overflow:hidden;text-overflow:ellipsis}
.status{display:inline-flex;border-radius:999px;padding:2px 8px;font-size:12px;background:var(--green-bg);color:var(--green)}
.status.off{background:var(--red-bg);color:var(--red)}
.status.warning{background:var(--amber-bg);color:var(--amber)}
.bar{width:100px;height:7px;background:#e7eae8;border-radius:999px;overflow:hidden}
.bar span{display:block;height:100%;background:var(--accent)}
.empty{padding:35px 16px;color:var(--muted);text-align:center}
.side{padding:3px 13px 12px}
.latency{padding:10px 0;border-bottom:1px solid var(--line)}
.latency:last-child{border:0}
.latency-head{display:flex;justify-content:space-between;gap:10px}
.latency-values{display:grid;grid-template-columns:repeat(3,minmax(72px,1fr));gap:8px;margin-top:8px}
.latency-value{background:#fafbfa;border:1px solid var(--line);border-radius:6px;padding:7px 8px}
.latency-value span{display:block;color:var(--muted);font-size:11px;margin-bottom:3px}
.latency-value strong{font:600 12px var(--mono)}
.muted{color:var(--muted)}
@media(max-width:1100px){.summary{grid-template-columns:repeat(3,minmax(125px,1fr))}.layout{grid-template-columns:1fr}}
@media(max-width:640px){header,main{padding-left:13px;padding-right:13px}.header-row{align-items:flex-start;flex-direction:column}.summary{grid-template-columns:repeat(2,minmax(125px,1fr))}.header-actions{flex-wrap:wrap}}
</style>
</head>
<body>
<header><div class="header-row">
<div class="title"><div id="dot" class="dot"></div><h1>InfiniLM Cluster Monitor</h1></div>
<div class="header-actions">
<span id="source" class="badge">frontend router</span>
<span id="updated">--</span>
<button id="auto" class="primary" aria-pressed="true">自动刷新</button>
<button id="refresh">刷新</button>
<a href="/metrics">Metrics</a>
</div>
</div></header>
<div id="error" class="error">监控数据获取失败</div>
<main>
<section class="summary">
<article class="metric"><div class="label">Workers</div><div id="workers" class="value">-</div><div id="minimum" class="sub">-</div></article>
<article class="metric"><div class="label">Inflight</div><div id="inflight" class="value">-</div><div class="sub">frontend 在途请求</div></article>
<article class="metric"><div class="label">Requests</div><div id="requests" class="value">-</div><div id="modes" class="sub">-</div></article>
<article class="metric"><div class="label">Worker requests</div><div id="worker-requests" class="value">-</div><div id="worker-active" class="sub">-</div></article>
<article class="metric"><div class="label">Usage tokens</div><div id="usage-tokens" class="value">-</div><div id="usage-parts" class="sub">-</div></article>
<article class="metric"><div class="label">Errors / Retries</div><div id="errors" class="value">-</div><div id="retries" class="sub">-</div></article>
<article class="metric"><div class="label">Token chunks</div><div id="tokens" class="value">-</div><div class="sub">流式 delta chunk</div></article>
<article class="metric"><div class="label">Direct / untracked</div><div id="direct" class="value">-</div><div class="sub">worker 收到减 frontend 路由</div></article>
<article class="metric"><div class="label">Affinity</div><div id="affinity-hits" class="value">-</div><div id="affinity-parts" class="sub">-</div></article>
<article class="metric"><div class="label">KV routing</div><div id="kv-aware-requests" class="value">-</div><div id="kv-routing-parts" class="sub">-</div></article>
<article class="metric"><div class="label">KV saved tokens</div><div id="kv-predicted-saved-tokens" class="value">-</div><div id="kv-actual-saved-tokens" class="sub">-</div></article>
</section>
<div class="layout">
<section class="panel"><div class="panel-header"><h2>Workers</h2><span id="mode" class="badge">-</span></div>
<div class="wrap"><table><thead><tr>
<th>Worker</th><th>Model</th><th>GPU</th><th>Endpoint</th><th>State</th><th class="num">Heartbeat</th><th class="num">Lease</th><th class="num">Load</th><th class="num">Frontend inflight</th><th class="num">Worker active</th><th class="num">Waiting</th><th class="num">Running</th><th>KV</th><th class="num">Cache hits</th><th class="num">Hit tokens</th><th class="num">Prefill tokens</th><th class="num">Routed</th><th class="num">Received</th><th class="num">Direct</th><th class="num">Errors</th><th class="num">Usage</th><th class="num">Total p95</th><th class="num">TTFT p95</th><th>Last success</th>
</tr></thead><tbody id="rows"></tbody></table></div><div id="empty" class="empty" hidden>No workers</div></section>
<section class="panel"><div class="panel-header"><h2>Latency</h2><span class="muted">avg / p95 / p99</span></div><div class="side">
<div class="latency"><div class="latency-head"><span>Request duration</span><strong id="duration-avg">-</strong></div><div class="latency-values"><div><span>p50</span><strong id="duration-p50">-</strong></div><div><span>p95</span><strong id="duration-p95">-</strong></div><div><span>p99</span><strong id="duration-p99">-</strong></div></div></div>
<div class="latency"><div class="latency-head"><span>Streaming TTFT</span><strong id="ttft-avg">-</strong></div><div class="latency-values"><div><span>p50</span><strong id="ttft-p50">-</strong></div><div><span>p95</span><strong id="ttft-p95">-</strong></div><div><span>p99</span><strong id="ttft-p99">-</strong></div></div></div>
<div class="latency"><div class="latency-head"><span>Inter-token</span><strong id="itl-avg">-</strong></div><div class="latency-values"><div><span>p50</span><strong id="itl-p50">-</strong></div><div><span>p95</span><strong id="itl-p95">-</strong></div><div><span>p99</span><strong id="itl-p99">-</strong></div></div></div>
</div></section>
</div>
</main>
<script>
const $=id=>document.getElementById(id);
const state={auto:true,timer:null,busy:false,lastGenerated:0,backoffMs:2000};
const number=v=>Number.isFinite(Number(v))?Number(v).toLocaleString("zh-CN"):"-";
const seconds=v=>{v=Number(v);if(!Number.isFinite(v)||v<=0)return "-";if(v<1)return (v*1000).toFixed(0)+" ms";if(v<60)return v.toFixed(2)+" s";return Math.floor(v/60)+"m "+Math.round(v%60)+"s"};
const clock=v=>Number(v)?new Date(v*1000).toLocaleTimeString("zh-CN",{hour12:false}):"-";
function cell(row,value,cls){const td=document.createElement("td");td.textContent=value??"-";if(cls)td.className=cls;row.appendChild(td);return td}
function workerRow(w){const tr=document.createElement("tr");const id=cell(tr,w.instance_id,"mono ellipsis");id.title=w.instance_id;const model=cell(tr,w.served_model_name,"ellipsis");model.title=w.model||w.served_model_name||"";cell(tr,w.gpu,"mono");cell(tr,w.endpoint,"mono");const health=document.createElement("td");const status=document.createElement("span");const draining=w.state==="draining";status.className=!w.engine_healthy?"status off":draining?"status warning":"status";status.textContent=!w.engine_healthy?"Unhealthy":draining?"Draining":"Ready";health.appendChild(status);tr.appendChild(health);cell(tr,w.heartbeat_age_seconds.toFixed(1)+"s","num");cell(tr,w.lease_remaining_seconds.toFixed(1)+"s","num");cell(tr,Number(w.load_score||0).toFixed(3),"num");cell(tr,number(w.inflight),"num");cell(tr,number(w.worker_active_requests),"num");cell(tr,number(w.waiting_requests),"num");cell(tr,number(w.running_requests)+"/"+number(w.max_batch_size),"num");const kv=document.createElement("td");const bar=document.createElement("div");bar.className="bar";const fill=document.createElement("span");fill.style.width=Math.max(0,Math.min(1,Number(w.kv_usage_ratio||0)))*100+"%";bar.appendChild(fill);kv.appendChild(bar);tr.appendChild(kv);cell(tr,number(w.cache_hit_requests_total),"num");cell(tr,number(w.cache_hit_tokens_total),"num");cell(tr,number(w.prefill_tokens_total),"num");cell(tr,number(w.requests_total),"num");cell(tr,number(w.worker_received_requests_total),"num");cell(tr,number(w.untracked_requests_total),"num");cell(tr,number(w.errors_total),"num");cell(tr,number(w.usage_total_tokens_total),"num");cell(tr,seconds(w.request_duration?.p95),"num");cell(tr,seconds(w.ttft?.p95),"num");cell(tr,clock(w.last_success_timestamp),"mono");return tr}
function render(data){state.lastGenerated=data.generated_at;$("dot").className="dot "+(data.ready?"ready":"warning");$("updated").textContent="更新 "+clock(data.generated_at);$("source").textContent=data.metrics_source;$("workers").textContent=number(data.worker_count);$("minimum").textContent="Ready "+number(data.worker_states?.ready??data.worker_count)+" / Draining "+number(data.worker_states?.draining??0)+"，最小副本 "+number(data.minimum_workers);$("inflight").textContent=number(data.global.inflight);$("requests").textContent=number(data.global.requests_total);$("modes").textContent="流式 "+number(data.global.streaming_requests_total)+" / 非流式 "+number(data.global.non_streaming_requests_total);$("worker-requests").textContent=number(data.global.worker_received_requests_total);$("worker-active").textContent="Active "+number(data.global.worker_active_requests);$("usage-tokens").textContent=number(data.global.usage_total_tokens_total);$("usage-parts").textContent="Prompt "+number(data.global.usage_prompt_tokens_total)+" / Completion "+number(data.global.usage_completion_tokens_total);$("errors").textContent=number(data.global.errors_total);$("retries").textContent="重试 "+number(data.global.retries_total);$("tokens").textContent=number(data.global.tokens_total);$("direct").textContent=number(data.global.untracked_requests_total);$("affinity-hits").textContent=number(data.global.affinity_hits_total);$("affinity-parts").textContent="Miss "+number(data.global.affinity_misses_total)+" / Entries "+number(data.global.affinity_entries);$("kv-aware-requests").textContent=number(data.global.kv_aware_requests_total);$("kv-routing-parts").textContent="Hit "+number(data.global.kv_hit_requests_total)+" / Miss "+number(data.global.kv_miss_requests_total)+" / Index "+number(data.global.kv_index_entries);$("kv-predicted-saved-tokens").textContent=number(data.global.kv_predicted_saved_tokens_total);$("kv-actual-saved-tokens").textContent="Actual "+number(data.global.kv_actual_saved_tokens_total);$("mode").textContent=data.router_mode;const d=data.latency.request_duration,t=data.latency.ttft,i=data.latency.inter_token_latency;for(const [prefix,h] of [["duration",d],["ttft",t],["itl",i]]){$(prefix+"-avg").textContent=seconds(h.avg);$(prefix+"-p50").textContent=seconds(h.p50);$(prefix+"-p95").textContent=seconds(h.p95);$(prefix+"-p99").textContent=seconds(h.p99)}const rows=$("rows");while(rows.firstChild)rows.removeChild(rows.firstChild);$("empty").hidden=data.workers.length>0;for(const worker of data.workers)rows.appendChild(workerRow(worker))}
function schedule(){if(state.timer){clearInterval(state.timer);state.timer=null}if(state.auto)state.timer=setInterval(load,state.backoffMs)}
async function load(){if(state.busy)return;state.busy=true;try{const response=await fetch("/api/monitoring",{cache:"no-store"});if(!response.ok)throw new Error("HTTP "+response.status);const data=await response.json();if(data.generated_at<state.lastGenerated){state.backoffMs=2000;return}render(data);state.backoffMs=2000;$("error").style.display="none"}catch(error){state.backoffMs=Math.min(30000,Math.max(2000,state.backoffMs*2));$("error").style.display="block";$("error").textContent="监控数据获取失败："+error.message}finally{state.busy=false;if(state.timer)schedule()}}
$("refresh").addEventListener("click",load);
$("auto").addEventListener("click",()=>{state.auto=!state.auto;$("auto").setAttribute("aria-pressed",String(state.auto));if(state.auto){state.backoffMs=2000;load()}schedule()});
document.addEventListener("visibilitychange",()=>{if(document.hidden&&state.timer){clearInterval(state.timer);state.timer=null}else if(!document.hidden&&state.auto){load();schedule()}});
load();schedule();
</script>
</body>
</html>'''

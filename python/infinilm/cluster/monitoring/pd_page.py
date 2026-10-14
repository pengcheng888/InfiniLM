"""Zero-dependency monitoring dashboard served by PdClusterFrontend."""

from __future__ import annotations

import time
from typing import Any

PD_MONITORING_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>InfiniLM PD Monitor</title>
<style>
:root{--bg:#f2f5f7;--panel:#fff;--text:#17222a;--muted:#64727c;--line:#d9e0e5;--accent:#0c6b76;--accent-bg:#e2f2f3;--green:#167347;--green-bg:#e7f4ec;--amber:#8a5300;--amber-bg:#fff1d7;--red:#a4262c;--red-bg:#fde9ea;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:14px 20px}
.header-row{max-width:1440px;margin:auto;display:flex;justify-content:space-between;align-items:center;gap:16px}
.title{display:flex;align-items:center;gap:10px;min-width:0} h1{font-size:18px;margin:0;letter-spacing:0}
.dot{width:10px;height:10px;border-radius:50%;background:var(--muted)} .dot.ready{background:var(--green)} .dot.down{background:var(--red)}
.header-actions{display:flex;align-items:center;gap:8px;color:var(--muted);white-space:nowrap}
.badge,button,a{border:1px solid var(--line);border-radius:6px}
.badge{display:inline-flex;align-items:center;padding:5px 8px;font-size:12px;color:var(--accent);background:var(--accent-bg);border-color:transparent}
button,a{padding:5px 9px;font:inherit;font-size:13px;color:var(--text);background:#fff;cursor:pointer;text-decoration:none}
button.primary{color:#fff;background:var(--accent);border-color:var(--accent)} button[aria-pressed="false"]{color:var(--muted)}
main{max-width:1440px;margin:auto;padding:18px 20px 32px} .error{display:none;margin:0 20px 14px;padding:9px 12px;border:1px solid #edc7ca;border-radius:7px;background:var(--red-bg);color:var(--red)}
.summary{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:10px;margin-bottom:14px}
.metric,.panel{background:var(--panel);border:1px solid var(--line);border-radius:7px}
.metric{padding:12px;min-height:78px} .label{color:var(--muted);font-size:12px;margin-bottom:7px} .value{font:600 21px/1 var(--mono);overflow-wrap:anywhere} .sub{margin-top:6px;color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.panel-header{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:11px 13px;border-bottom:1px solid var(--line)} h2{font-size:14px;margin:0}
.wrap{overflow-x:auto} table{width:100%;min-width:620px;border-collapse:collapse} th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;font-size:13px} th{color:var(--muted);font-size:12px;background:#fafbfc} th.num,td.num{text-align:right} td.mono{font-family:var(--mono);font-size:12px}
.status{display:inline-flex;border-radius:999px;padding:2px 8px;font-size:12px;background:var(--green-bg);color:var(--green)} .status.off{background:var(--red-bg);color:var(--red)} .status.draining{background:var(--amber-bg);color:var(--amber)}
.empty{padding:30px 14px;color:var(--muted);text-align:center} .footer{margin-top:10px;color:var(--muted);font-size:12px;text-align:right}
@media(max-width:1100px){.summary{grid-template-columns:repeat(3,minmax(130px,1fr))}.grid{grid-template-columns:1fr}}
@media(max-width:680px){header,main{padding-left:12px;padding-right:12px}.header-row{align-items:flex-start;flex-direction:column}.summary{grid-template-columns:repeat(2,minmax(130px,1fr))}.header-actions{flex-wrap:wrap}}
</style>
</head>
<body>
<header><div class="header-row">
<div class="title"><div id="dot" class="dot"></div><h1>InfiniLM PD Monitor</h1></div>
<div class="header-actions"><span class="badge">PD frontend</span><span id="updated">--</span><button id="auto" class="primary" aria-pressed="true">自动刷新</button><button id="refresh">刷新</button><a href="/metrics">Metrics</a></div>
</div></header>
<div id="error" class="error">监控数据获取失败</div>
<main>
<section class="summary">
<div class="metric"><div class="label">状态</div><div id="ready" class="value">--</div><div id="model" class="sub">--</div></div>
<div class="metric"><div class="label">Prefill</div><div id="prefill-count" class="value">0</div><div id="prefill-state" class="sub">--</div></div>
<div class="metric"><div class="label">Decode</div><div id="decode-count" class="value">0</div><div id="decode-state" class="sub">--</div></div>
<div class="metric"><div class="label">P 请求</div><div id="prefill-requests" class="value">0</div><div class="sub">selected</div></div>
<div class="metric"><div class="label">D 请求</div><div id="decode-requests" class="value">0</div><div class="sub">selected</div></div>
</section>
<section class="grid">
<div class="panel"><div class="panel-header"><h2>Prefill Workers</h2><span id="prefill-summary" class="badge">0 ready</span></div><div class="wrap"><table><thead><tr><th>Worker</th><th>GPU</th><th>Endpoint</th><th>Engine ID</th><th>State</th><th class="num">Inflight</th><th class="num">Waiting</th><th class="num">Running</th><th class="num">KV</th></tr></thead><tbody id="prefill-workers"></tbody></table><div id="prefill-empty" class="empty">No prefill workers</div></div></div>
<div class="panel"><div class="panel-header"><h2>Decode Workers</h2><span id="decode-summary" class="badge">0 ready</span></div><div class="wrap"><table><thead><tr><th>Worker</th><th>GPU</th><th>Endpoint</th><th>Engine ID</th><th>State</th><th class="num">Inflight</th><th class="num">Waiting</th><th class="num">Running</th><th class="num">KV</th></tr></thead><tbody id="decode-workers"></tbody></table><div id="decode-empty" class="empty">No decode workers</div></div></div>
</section>
<div id="footer" class="footer">--</div>
</main>
<script>
const $=id=>document.getElementById(id);
let automatic=true;
function text(value,fallback='--'){return value===null||value===undefined||value===''?fallback:String(value)}
function esc(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function stateLabel(state){return state==='ready'?'ready':state==='draining'?'drain':'down'}
function workerRow(worker){
 const load=worker.load||{}; const kv=Number(load.kv_usage_ratio||0)*100;
 return `<tr><td class="mono" title="${esc(worker.instance_id)}">${esc(worker.short_id)}</td><td>${esc(text(worker.gpu))}</td><td class="mono">${esc(worker.endpoint)}</td><td class="mono">${esc(text(worker.engine_id))}</td><td><span class="status ${worker.state==='ready'?'':'off'}">${stateLabel(worker.state)}</span></td><td class="num mono">${worker.inflight}</td><td class="num mono">${load.waiting_requests??0}</td><td class="num mono">${load.running_requests??0}</td><td class="num mono">${kv.toFixed(1)}%</td></tr>`
}
function render(data){
 const ready=Boolean(data.ready); $('dot').className='dot '+(ready?'ready':'down'); $('ready').textContent=ready?'Ready':'Not ready';
 $('model').textContent=(data.models||[]).join(', ')||'no model'; $('updated').textContent=new Date((data.generated_at||0)*1000).toLocaleTimeString();
 for(const role of ['prefill','decode']){
  const workers=(data.workers||[]).filter(item=>item.role===role); const readyCount=workers.filter(item=>item.state==='ready').length;
  $(role+'-count').textContent=String(workers.length); $(role+'-summary').textContent=`${readyCount} ready`; $(role+'-state').textContent=`${readyCount}/${workers.length} ready`;
  $(role+'-workers').innerHTML=workers.map(workerRow).join(''); $(role+'-empty').style.display=workers.length?'none':'block';
  $(role+'-requests').textContent=String((data.requests_total||{}).prefill??0);
 }
 $('decode-requests').textContent=String((data.requests_total||{}).decode??0);
 $('footer').textContent=`namespace ${text(data.namespace)} · router ${text(data.router_mode)}`;
}
async function refresh(){
 try{const response=await fetch('/api/monitoring',{cache:'no-store'});if(!response.ok)throw new Error(response.statusText);render(await response.json());$('error').style.display='none'}
 catch(error){$('error').style.display='block';$('error').textContent=`监控数据获取失败：${error.message}`}
}
$('refresh').onclick=refresh;
$('auto').onclick=()=>{automatic=!automatic;$('auto').setAttribute('aria-pressed',String(automatic));$('auto').classList.toggle('primary',automatic);if(automatic)refresh()};
refresh();setInterval(()=>{if(automatic)refresh()},2000);
</script>
</body>
</html>'''


def _worker_snapshot(frontend: Any, worker: Any) -> dict[str, Any]:
    metadata = worker.metadata or {}
    load = metadata.get("load") if isinstance(metadata.get("load"), dict) else {}
    pd_metadata = metadata.get("pd") if isinstance(metadata.get("pd"), dict) else {}
    return {
        "role": worker.worker_type,
        "instance_id": worker.instance_id,
        "short_id": worker.instance_id[-12:],
        "served_model_name": worker.served_model_name,
        "endpoint": f"{worker.host}:{worker.port}",
        "gpu": metadata.get("cuda_visible_devices"),
        "engine_id": pd_metadata.get("engine_id"),
        "capabilities": list(metadata.get("capabilities") or []),
        "needs": list(metadata.get("needs") or []),
        "state": worker.state,
        "engine_healthy": bool(worker.engine_healthy),
        "inflight": frontend.inflight.get(worker.instance_id, 0),
        "heartbeat_age_seconds": max(0.0, time.time() - worker.heartbeat_at),
        "lease_remaining_seconds": max(0.0, worker.lease_expires_at - time.time()),
        "load": load,
    }


def pd_monitoring_snapshot(frontend: Any) -> dict[str, Any]:
    workers = frontend.watcher.workers
    roles = ("prefill", "decode")
    return {
        "generated_at": time.time(),
        "namespace": frontend.options.namespace,
        "router_mode": frontend.options.router_mode,
        "request_plane": frontend.options.request_plane,
        "models": [model for model in frontend._ready_models()],
        "ready": any(frontend._is_model_ready(model) for model in frontend._ready_models()),
        "workers": [_worker_snapshot(frontend, worker) for worker in workers],
        "requests_total": {
            role: frontend.request_counters.get(role, 0) for role in roles
        },
        "minimum_workers": {
            "prefill": frontend.options.min_prefill_workers,
            "decode": frontend.options.min_decode_workers,
        },
    }

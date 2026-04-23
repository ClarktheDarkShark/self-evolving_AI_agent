import argparse
import json
import os
import yaml
import copy
from enum import StrEnum
from typing import Any, Mapping, Sequence, Optional
import coredumpy  # type: ignore[import-untyped]

from src.utils import ConfigLoader, SingletonLogger
from src.utils.output_paths import get_output_dir_override, prefix_filename
from src.typings import (
    AssignmentConfig,
    EnvironmentConfig,
    SampleStatus,
    LoggerConfig,
    ContinualAgentBenchException,
    Session,
    SessionEvaluationOutcome,
    SampleIndex,
    PathConfig,
    GeneralInstanceFactory,
    SessionMetricCalculationPartial,
    ChatHistoryItem,
    Role,
)
from src.tasks import Task, DatasetItem
from src.agents import Agent
from src.language_models import LanguageModel
from src.callbacks import (
    CallbackHandler,
    CallbackConstructor,
    Callback,
    CallbackRestorer,
    CallbackArguments,
)


# Toggle for post-task Reflection Forge.  When False the automatic
# generate_tool_from_failure() call after each incorrect/limit-reached
# task is skipped.  The mid-run Orchestrator escape-hatch
# (request_new_tool) remains fully operational regardless of this flag.
ENABLE_POST_TASK_REFLECTION = False
ENABLE_SAGE_AGENT = os.environ.get("ENABLE_SAGE_AGENT") == "1"
SAGE_AGENT_NAME = "sage_agent_controller"
SAGE_AGENT_MAX_COMPLETION_TOKENS_ENV = "SAGE_AGENT_MAX_COMPLETION_TOKENS"
SAGE_AGENT_REASONING_EFFORT_ENV = "SAGE_AGENT_REASONING_EFFORT"
LANGUAGE_MODEL_OVERRIDE_ENV = "LIFELONG_MODEL"
LANGUAGE_MODEL_OVERRIDE_FALLBACK_ENV = "OPENAI_MODEL"
SAGE_AGENT_COMPONENT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs",
    "components",
    "agents",
    "sage_agent_controller.yaml",
)

# ---------------------------------------------------------------------------
# HTML Trace Viewer
# ---------------------------------------------------------------------------
_HTML_TRACE_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Trace Viewer</title>
<script>
/* Live Server smart-reload intercept — patches window.WebSocket before livereload.js runs */
(function(){
  var _WS=window.WebSocket;
  var _timer=null;
  function _smartReload(){
    clearTimeout(_timer);
    _timer=setTimeout(function(){
      var base=location.href.replace(/\\?.*$/,'').replace(/\\/[^\\/]*$/,'/');
      fetch(base+'sessions.json?_t='+Date.now())
        .then(function(r){return r.json();})
        .then(function(nd){
          if(!Array.isArray(nd))return;
          if(nd.length<=(window._liveData||[]).length)return;
          window._liveData=nd;
          window._liveRefresh&&window._liveRefresh(nd);
        })
        .catch(function(){});
    },250);
  }
  function wrapMsg(fn,ws){
    return function(evt){
      try{var d=JSON.parse(evt.data);if(d&&d.command==='reload'){_smartReload();return;}}catch(e){}
      fn&&fn.call(ws,evt);
    };
  }
  function PatchedWS(url,proto){
    var ws=proto!==undefined?new _WS(url,proto):new _WS(url);
    if(!/35729|livereload/i.test(String(url)))return ws;
    var _omcb=null;
    Object.defineProperty(ws,'onmessage',{configurable:true,
      get:function(){return _omcb;},
      set:function(fn){if(_omcb)ws.removeEventListener('message',_omcb);_omcb=wrapMsg(fn,ws);ws.addEventListener('message',_omcb);}
    });
    var _ael=ws.addEventListener.bind(ws);
    ws.addEventListener=function(type,fn,opts){_ael(type,type==='message'?wrapMsg(fn,ws):fn,opts);};
    return ws;
  }
  PatchedWS.prototype=_WS.prototype;
  try{Object.keys(_WS).forEach(function(k){PatchedWS[k]=_WS[k];});}catch(e){}
  window.WebSocket=PatchedWS;
})();
</script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;display:flex;height:100vh;overflow:hidden;background:#12121f;color:#dde1f0}

/* ── Sidebar ── */
#sidebar{width:288px;overflow:hidden;background:#161626;border-right:1px solid #252540;flex-shrink:0;display:flex;flex-direction:column}
#sb-head{padding:12px 14px 10px;background:#161626;border-bottom:1px solid #252540;flex-shrink:0}
#sb-title{font-size:11px;color:#6870a0;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:8px}
#sb-stats{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap}
.stat-pill{padding:2px 9px;border-radius:12px;font-size:11px;font-weight:600}
.sp-c{background:#162a1a;color:#6fcf7c;border:1px solid #2a5c35}
.sp-i{background:#2a1616;color:#f07070;border:1px solid #5c2a2a}
.sp-u{background:#1e1e35;color:#8890a8;border:1px solid #252540}
#sb-search{width:100%;padding:6px 10px;background:#1a1a2e;border:1px solid #252540;border-radius:6px;color:#dde1f0;font-size:12px;outline:none;margin-bottom:8px}
#sb-search:focus{border-color:#4d6aff}
#sb-search::placeholder{color:#44445a}
#sb-filters{display:flex;gap:4px;flex-wrap:wrap}
.flt{padding:3px 9px;border-radius:12px;font-size:11px;cursor:pointer;border:1px solid #252540;background:transparent;color:#6870a0;transition:all .12s}
.flt.active,.flt:hover{background:#1e2240;color:#dde1f0;border-color:#4d6aff}
#sl{overflow-y:auto;flex:1}
.si{padding:10px 14px;cursor:pointer;border-bottom:1px solid #1a1a2e;transition:background .12s}
.si:hover{background:#1a1a2e}
.si.active{background:#1a2240;border-left:3px solid #4d6aff}
.si-row1{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.si-idx{font-size:11px;color:#6870a0}
.si-badges{display:flex;gap:4px;align-items:center}
.si-oc{font-size:11px;font-weight:700}
.oc-correct{color:#6fcf7c}.oc-incorrect{color:#f07070}.oc-unset{color:#8890a8}
.si-sage{font-size:9px;background:#0d1e30;color:#5ba8ff;border:1px solid #1d4060;border-radius:3px;padding:1px 5px;letter-spacing:.5px}
.si-q{font-size:11px;color:#6878a0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* ── Main panel ── */
#main{flex:1;overflow-y:auto;padding:22px 28px}
#empty{display:flex;align-items:center;justify-content:center;height:100%;color:#2a2a4a;font-size:18px}

/* ── Answer summary card ── */
.ans-card{border-radius:10px;padding:16px 20px;margin-bottom:20px;border:1px solid}
.ans-card.oc-correct{background:linear-gradient(135deg,#0a1810,#0d1622);border-color:#2a5c35}
.ans-card.oc-incorrect{background:linear-gradient(135deg,#1a0a0a,#150c18);border-color:#5c2a2a}
.ans-card.oc-unset{background:#161626;border-color:#252540}
.ac-top{display:flex;align-items:center;gap:12px;margin-bottom:10px}
.ac-outcome{font-size:17px;font-weight:700;letter-spacing:.3px}
.ac-outcome.oc-correct{color:#6fcf7c}
.ac-outcome.oc-incorrect{color:#f07070}
.ac-outcome.oc-unset{color:#8890a8}
.ac-f1{font-size:11px;color:#8890a8;background:#1a1a2e;padding:3px 8px;border-radius:4px}
.ac-si{font-size:11px;color:#6870a0;margin-left:auto}
.ac-q{font-size:14px;color:#dde1f0;line-height:1.55;margin-bottom:12px;font-weight:500}
.ac-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.ac-box{background:#1a1a2e;border-radius:8px;padding:10px 14px}
.ac-box-agent{border-left:3px solid #4d6aff}
.ac-box-expected{border-left:3px solid #6fcf7c}
.ac-box-label{font-size:9px;text-transform:uppercase;letter-spacing:1.2px;color:#44547a;margin-bottom:5px}
.ac-box-val{font-size:13px;font-weight:600;color:#dde1f0;line-height:1.4}
.ac-box-mid{font-size:10px;color:#44547a;margin-top:3px;font-family:monospace}

/* ── Pipeline stage timeline ── */
.stage-wrap{display:flex;align-items:flex-start;gap:0;margin-bottom:20px;overflow-x:auto;padding:4px 0 8px}
.stage{display:flex;flex-direction:column;align-items:center;min-width:80px;max-width:96px}
.stage-conn{height:2px;flex:1;min-width:10px;max-width:28px;margin-top:15px;flex-shrink:0}
.stage-conn.s-done{background:linear-gradient(90deg,#4d6aff,#4d6aff)}
.stage-conn.s-pending{background:#252540}
.stage-dot{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;border:2px solid;margin-bottom:5px}
.stage-dot.s-done{background:#0a1428;border-color:#4d6aff}
.stage-dot.s-warn{background:#1e1408;border-color:#f0a030}
.stage-dot.s-fail{background:#1e0808;border-color:#f07070}
.stage-dot.s-neutral{background:#1a1a2e;border-color:#252540}
.stage-lbl{font-size:10px;color:#8890a8;text-align:center;line-height:1.3}
.stage-det{font-size:9px;color:#5ba8ff;text-align:center;margin-top:2px;line-height:1.3}
.stage-det.s-warn{color:#f0a030}

/* ── SAGE execution summary ── */
.sage-card{background:#0a1828;border:1px solid #1d3050;border-radius:8px;padding:14px 16px;margin-bottom:20px}
.sage-card-title{font-size:10px;color:#5ba8ff;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:12px;display:flex;align-items:center;gap:6px}
.sage-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.sage-stat{background:#0a1220;border-radius:6px;padding:9px 11px;border:1px solid #152030}
.sage-stat-k{font-size:9px;text-transform:uppercase;letter-spacing:1px;color:#3a5070;margin-bottom:4px}
.sage-stat-v{font-size:13px;font-weight:600;color:#c8d8f0}
.sage-stat-v.pv-good{color:#6fcf7c}
.sage-stat-v.pv-empty{color:#f0a030}
.sage-stat-v.pv-mono{font-family:monospace;font-size:12px}

/* ── Conversation ── */
.sec{margin-bottom:22px}
.sec-title{font-size:10px;color:#44547a;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:10px}
.turn{margin-bottom:8px;display:flex;gap:8px;align-items:flex-start}
.turn-user{flex-direction:row}
.turn-agent{flex-direction:row-reverse}
.t-icon{width:26px;height:26px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:12px;margin-top:1px}
.turn-user .t-icon{background:#1a2240;color:#5ba8ff}
.turn-agent .t-icon{background:#142014;color:#6fcf7c}
.t-body{max-width:88%;display:flex;flex-direction:column;gap:3px}
.turn-agent .t-body{align-items:flex-end}
.t-role{font-size:9px;color:#3a3a5a}
.bubble{padding:8px 12px;border-radius:8px;font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word;color:#c0c8e8}
.turn-user .bubble{background:#181830;border:1px solid #252540}
.turn-agent .bubble{background:#121c12;border:1px solid #1e2e1e;color:#b0d8b0}

/* Prompt collapse */
.prompt-hdr{background:#181830;border:1px solid #252540;border-radius:6px;padding:7px 12px;font-size:11px;color:#44547a;cursor:pointer;display:flex;align-items:center;gap:6px;user-select:none;width:100%}
.prompt-hdr:hover{color:#8890a8}
.prompt-body{display:none;margin-top:3px;background:#181830;border:1px solid #252540;border-radius:6px;padding:10px 12px;font-size:11px;color:#6878a0;white-space:pre-wrap;word-break:break-word;max-height:180px;overflow-y:auto;line-height:1.5}
.prompt-body.open{display:block}

/* Question block */
.q-block{background:#181830;border:1px solid #252540;border-radius:8px;padding:10px 14px;border-left:3px solid #4d6aff}
.q-lbl{font-size:9px;text-transform:uppercase;letter-spacing:1px;color:#4d6aff;margin-bottom:4px}
.q-text{font-size:13px;color:#dde1f0;line-height:1.5;font-weight:500}
.q-ents{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}
.ent-chip{background:#1a2a3a;border:1px solid #2a3a5a;border-radius:12px;padding:2px 8px;font-size:10px;color:#5ba8ff}

/* Action chips */
.act-chip{display:inline-flex;align-items:center;gap:5px;background:#0e1828;border:1px solid #1d2e40;border-radius:6px;padding:5px 10px;font-family:monospace;font-size:11px;color:#a8c4e0;max-width:100%}
.act-fn{color:#7ab4ff;font-weight:600}
.act-arg{color:#f0c060}
.act-rel{color:#88d888;font-size:10px;word-break:break-all}
.act-op{color:#c88af0}

/* Bridge block */
.bridge-block{background:#081828;border:1px solid #1d4060;border-radius:10px;padding:12px 14px;border-left:3px solid #5ba8ff}
.bridge-block.bridge-ok{border-left-color:#6fcf7c}
.bridge-block.bridge-warn{border-left-color:#f0a030}
.bridge-block.bridge-fail{border-left-color:#f07070}
.bridge-top{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:10px}
.bridge-head{display:flex;align-items:center;gap:8px;min-width:0}
.bridge-title-wrap{display:flex;flex-direction:column;gap:2px;min-width:0}
.bridge-title{font-size:12px;font-weight:700;color:#dde7ff}
.bridge-subtitle{font-size:10px;color:#6f87ad;line-height:1.4}
.bridge-badges{display:flex;flex-wrap:wrap;gap:5px;justify-content:flex-end}
.bridge-badge,.atype-chip{background:#0e2035;border:1px solid #1d4060;border-radius:999px;padding:2px 8px;font-size:10px;color:#7ab4ff;white-space:nowrap}
.bridge-badge.bridge-good{border-color:#2a5c35;background:#102015;color:#8cda97}
.bridge-badge.bridge-warn{border-color:#5c4620;background:#231707;color:#f0c060}
.bridge-badge.bridge-bad{border-color:#5c2a2a;background:#220d0d;color:#f09090}
.bridge-primary{background:#091321;border:1px solid #15263d;border-radius:8px;padding:10px 12px;margin-bottom:10px}
.bridge-primary-k{font-size:9px;text-transform:uppercase;letter-spacing:1px;color:#52719b;margin-bottom:4px}
.bridge-primary-v{font-size:16px;font-weight:700;color:#edf3ff;line-height:1.35;word-break:break-word}
.bridge-body{display:flex;flex-direction:column;gap:8px}
.bridge-note{display:flex;gap:8px;align-items:flex-start;font-size:11px;line-height:1.5}
.bridge-note-k{color:#6f87ad;min-width:112px;flex-shrink:0}
.bridge-note-v{color:#d4def2}
.bridge-paths{display:flex;flex-direction:column;gap:5px}
.bridge-path{background:#0b1320;border:1px solid #152030;border-radius:6px;padding:7px 9px;font-size:11px;line-height:1.45;color:#c7d5ec}
.bridge-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.bridge-stat{background:#0a1220;border-radius:6px;padding:8px 10px;border:1px solid #152030}
.bridge-stat-k{font-size:9px;text-transform:uppercase;letter-spacing:1px;color:#48617f;margin-bottom:4px}
.bridge-stat-v{font-size:12px;font-weight:600;color:#d6e3f8;word-break:break-word}
.bridge-stat-v.bridge-mono{font-family:monospace}
.bridge-raw{background:#0a1220;border:1px solid #152030;border-radius:6px;padding:8px 10px}
.bridge-raw-k{font-size:9px;text-transform:uppercase;letter-spacing:1px;color:#48617f;margin-bottom:4px}
.bridge-raw-v{font-size:11px;color:#c7d5ec;white-space:pre-wrap;word-break:break-word;line-height:1.5}
.bridge-raw-v.bridge-mono{font-family:monospace}

/* Final answer block */
.final-block{background:#0a1c0a;border:1px solid #2a5c35;border-radius:8px;padding:9px 14px;border-left:3px solid #6fcf7c;display:flex;align-items:center;gap:10px}
.final-icon{font-size:15px}
.final-text{font-size:13px;font-weight:600;color:#6fcf7c}

/* Observation block */
.obs-block{background:#161620;border:1px solid #222230;border-radius:6px;padding:8px 12px;width:100%}
.obs-act{font-size:10px;color:#44547a;margin-bottom:4px;font-family:monospace}
.obs-body{font-size:11px;color:#7880a0;white-space:pre-wrap;word-break:break-word;max-height:90px;overflow:hidden;line-height:1.4}
.obs-more{font-size:10px;color:#4d6aff;cursor:pointer;margin-top:3px;display:inline-block}
.obs-body.expanded{max-height:none}

/* Macro result */
.macro-res{padding:6px 12px;border-radius:6px;font-size:11px;font-weight:600;display:inline-flex;align-items:center;gap:6px}
.macro-ok{background:#0a1c0a;color:#6fcf7c;border:1px solid #2a5c35}
.macro-fail{background:#1c0a0a;color:#f07070;border:1px solid #5c2a2a}

/* Ack chip */
.ack{background:#1a1a2e;color:#44547a;padding:4px 10px;border-radius:4px;font-size:11px;display:inline-block}
</style></head>
<body>
<div id="sidebar">
  <div id="sb-head">
    <div id="sb-title">Sessions <span id="cnt"></span></div>
    <div id="sb-stats"></div>
    <input id="sb-search" type="search" placeholder="⌕  Search questions…">
    <div id="sb-filters">
      <button class="flt active" data-f="all">All</button>
      <button class="flt" data-f="correct">✅ Correct</button>
      <button class="flt" data-f="incorrect">❌ Incorrect</button>
      <button class="flt" data-f="sage">🌉 SAGE</button>
    </div>
  </div>
  <div id="sl"></div>
</div>
<div id="main"><div id="empty">← Select a session</div></div>
<script>
let logData=window._liveData=/*LOG_DATA_PLACEHOLDER*/;
const STATE_KEY='traceViewerState:'+location.pathname;
const KG_PROMPT_PFX="You are an intelligent agent tasked with answering questions by querying a knowledge base.";

// ── Utilities ──────────────────────────────────────────────────────────────
function esc(s){return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function ocClass(o){return o==='correct'?'correct':o==='incorrect'?'incorrect':'unset';}
function readState(){try{return JSON.parse(localStorage.getItem(STATE_KEY)||'{}')||{};}catch{return{};}}
function writeState(p){try{const c=readState();localStorage.setItem(STATE_KEY,JSON.stringify({...c,...p}));}catch{}}
function findIdx(si){const t=String(si??'').trim();return logData.findIndex(s=>String(s.sample_index??'').trim()===t);}
function readHashSI(){const p='#sample=';if(!location.hash.startsWith(p))return'';try{return decodeURIComponent(location.hash.slice(p.length));}catch{return location.hash.slice(p.length);}}
function setHash(si){const t='#sample='+encodeURIComponent(String(si??''));try{if(location.hash!==t)history.replaceState(null,'',t);}catch{location.hash=t;}}
function persistScroll(){writeState({sTop:document.getElementById('sidebar')?.scrollTop||0,mTop:document.getElementById('main')?.scrollTop||0});}
function restoreScroll(){const st=readState();const sb=document.getElementById('sidebar');const m=document.getElementById('main');if(sb&&st.sTop!=null)sb.scrollTop=st.sTop;if(m&&st.mTop!=null)m.scrollTop=st.mTop;}

// ── Data helpers ───────────────────────────────────────────────────────────
function hasBridge(s){return(s.chat_history?.value||[]).some(t=>(t.content||'').includes('sage_benchmark_bridge_macro'));}
function extractQuestion(turns){
  for(const t of turns){
    if(t.role!=='user')continue;
    const m=(t.content||'').match(/Question:\\s*(.*?)(?=,\\s*Entities:|$)/s);
    if(m)return m[1].trim();
  }
  return null;
}
function parseBridgePayload(turns){
  for(const t of turns){
    const c=t.content||'';
    if(!c.includes('sage_benchmark_bridge_macro'))continue;
    const m=c.match(/execute_macro\\([^,]+,\\s*(\\{[\\s\\S]*\\})\\)/);
    if(m){try{return JSON.parse(m[1]);}catch{}}
    break;
  }
  return null;
}

function humanizeToken(s){
  const txt=String(s??'').trim();
  if(!txt)return'—';
  return txt
    .replace(/[_-]+/g,' ')
    .replace(/\b\w/g,m=>m.toUpperCase());
}

function artifactMeta(type){
  const map={
    count_scalar:{icon:'#',label:'Count result',headline:'Final count',valueLabel:'Count'},
    entity_id:{icon:'🏷️',label:'Single entity',headline:'Selected entity',valueLabel:'Entity ID'},
    entity_set:{icon:'🗂️',label:'Entity set',headline:'Returned set',valueLabel:'Entity IDs'},
    scalar_literal:{icon:'🔢',label:'Literal value',headline:'Literal value',valueLabel:'Value'},
    text_literal:{icon:'📝',label:'Text value',headline:'Text value',valueLabel:'Value'},
    empty:{icon:'∅',label:'Empty result',headline:'Empty result',valueLabel:'Value'},
  };
  return map[type]||{icon:'🧩',label:humanizeToken(type||'result'),headline:'Result',valueLabel:'Value'};
}

function sourceLabel(source){
  const map={raw_execution:'Direct SAGE query result'};
  return map[source]||humanizeToken(source||'result source');
}

function asList(value){
  if(Array.isArray(value))return value.filter(v=>v!=null&&String(v).trim()!=='').map(v=>String(v));
  if(value==null)return[];
  const txt=String(value).trim();
  return txt?[txt]:[];
}

function splitRelationSummary(value){
  if(Array.isArray(value))return value.filter(Boolean).map(v=>String(v));
  const txt=String(value??'').trim();
  if(!txt)return[];
  return txt.split(/\s+\|\s+/).map(v=>v.trim()).filter(Boolean);
}

function formatPreviewValue(value){
  if(Array.isArray(value))return value.map(v=>String(v)).join(', ');
  if(value&&typeof value==='object')return JSON.stringify(value);
  const txt=String(value??'').trim();
  return txt||'—';
}

function formatRowPreview(rows){
  const items=Array.isArray(rows)?rows:[];
  if(!items.length)return'';
  return items.slice(0,3).map(row=>{
    if(row&&typeof row==='object'&&!Array.isArray(row)){
      return Object.entries(row).map(([k,v])=>`${k}=${v}`).join('; ');
    }
    return String(row);
  }).join(' | ');
}

function parseResolvedPreview(value){
  const txt=String(value??'').trim();
  if(!txt)return null;
  const m=txt.match(/^(.+?)\s*=\s*(.+)$/);
  return m?{raw:m[1].trim(),label:m[2].trim()}:null;
}

function deriveResolvedPreview(info){
  const direct=parseResolvedPreview(info.resolvedValuePreview);
  if(direct)return direct;
  const rows=Array.isArray(info.rowPreview)?info.rowPreview:[];
  for(const row of rows){
    if(!row||typeof row!=='object'||Array.isArray(row))continue;
    const keys=Object.keys(row);
    const nameKey=keys.find(k=>/_name$/i.test(k));
    if(!nameKey)continue;
    const valueKey=(info.projectedVar&&row[info.projectedVar]!=null)?info.projectedVar:keys.find(k=>k!==nameKey&&row[k]!=null);
    if(!valueKey)continue;
    return{raw:String(row[valueKey]),label:String(row[nameKey])};
  }
  return null;
}

function parseMacroResultText(text){
  const lines=String(text??'').split(String.fromCharCode(10)).map(line=>line.trim()).filter(Boolean);
  const header=lines[0]||'';
  const m=header.match(/^Macro result:\s*(\S+)\s*->\s*(\w+)/);
  if(!m)return null;
  const fields={};
  for(const line of lines.slice(1)){
    const idx=line.indexOf(':');
    if(idx<0)continue;
    const key=line.slice(0,idx).trim();
    const value=line.slice(idx+1).trim();
    fields[key]=value;
  }
  return{macroName:m[1],status:m[2],fields};
}

function normalizeFlag(value){
  const txt=String(value??'').trim().toLowerCase();
  if(txt==='yes'||txt==='true')return'yes';
  if(txt==='no'||txt==='false')return'no';
  return'';
}

function buildBridgeInfoFromPayload(payload){
  const diag=payload?.sage_artifact_diagnostics||{};
  return{
    artifactType:payload?.sage_artifact_type||'',
    primaryValue:payload?.sage_artifact_value,
    source:payload?.sage_artifact_source||'',
    semantic:payload?.sage_semantic_description||'',
    selectionBasis:payload?.sage_selection_basis||'',
    relationSummary:payload?.sage_relation_summary||[],
    projectedVar:payload?.sage_selected_query_variable||diag.selected_var||'',
    bindingCount:payload?.sage_binding_count??diag.binding_count,
    uniqueValueCount:payload?.sage_unique_value_count??diag.normalized_value_count,
    valuePreview:payload?.sage_value_preview??diag.value_preview,
    rowPreview:payload?.sage_row_preview??diag.row_preview,
    resolvedValuePreview:payload?.sage_resolved_value_preview||'',
    answerCardinality:payload?.sage_answer_cardinality_hint||'',
    completenessHint:payload?.sage_completeness_hint||'',
    proofHint:payload?.sage_proof_hint||'',
    repairCaveat:payload?.sage_repair_caveat||'',
    failureReason:payload?.sage_failure_reason||'',
    trustedFinal:normalizeFlag(payload?.sage_trusted_for_materialization),
    solvesTask:normalizeFlag(payload?.sage_solves_task),
    confidence:payload?.sage_confidence,
    finalVariable:'',
    status:String(payload?.sage_tool_status||'success'),
    observation:'',
  };
}

function buildBridgeInfoFromMacroText(text){
  const parsed=parseMacroResultText(text);
  if(!parsed||parsed.macroName!=='sage_benchmark_bridge_macro')return null;
  const f=parsed.fields;
  return{
    artifactType:f['Artifact type']||'',
    primaryValue:f['Value preview']||'',
    source:'',
    semantic:f['Semantic']||'',
    selectionBasis:f['Selection basis']||'',
    relationSummary:f['Relation summary']||'',
    projectedVar:f['Projected query variable']||'',
    bindingCount:f['Raw binding count']||'',
    uniqueValueCount:f['Unique value count']||'',
    valuePreview:f['Value preview']||'',
    rowPreview:f['Row preview']||'',
    resolvedValuePreview:f['Resolved value preview']||'',
    answerCardinality:f['Expected answer cardinality']||'',
    completenessHint:f['Completeness hint']||'',
    proofHint:f['Proof hint']||'',
    repairCaveat:f['Repair caveat']||'',
    failureReason:f['Failure reason']||'',
    trustedFinal:normalizeFlag(f['Trusted final']),
    solvesTask:normalizeFlag(f['Solves task']),
    confidence:f['Confidence']||'',
    finalVariable:f['Final variable']||'',
    status:String(parsed.status||'').toLowerCase(),
    observation:f['Observation']||'',
  };
}

function summarizePrimaryValue(info){
  const meta=artifactMeta(info.artifactType);
  const rawValue=formatPreviewValue(info.primaryValue);
  const resolved=deriveResolvedPreview(info);
  if(info.artifactType==='count_scalar')return{headline:rawValue,subline:meta.headline};
  if(info.artifactType==='entity_id'){
    if(resolved)return{headline:resolved.label,subline:meta.headline};
    return{headline:rawValue,subline:meta.valueLabel};
  }
  if(info.artifactType==='entity_set'){
    const count=info.uniqueValueCount!=null&&String(info.uniqueValueCount)!==''?String(info.uniqueValueCount):String(asList(info.valuePreview).length||'—');
    return{headline:`${count} value${count==='1'?'':'s'}`,subline:meta.headline};
  }
  if(info.artifactType==='empty')return{headline:'No value returned',subline:meta.headline};
  return{headline:rawValue,subline:meta.headline};
}

function renderBridgeInfoCard(info,mode){
  if(!info)return'';
  const meta=artifactMeta(info.artifactType);
  const primary=summarizePrimaryValue(info);
  const status=String(info.status||'').toLowerCase();
  const ok=status==='success';
  const trusted=info.trustedFinal==='yes';
  const solves=info.solvesTask==='yes';
  const variant=ok?(trusted?'bridge-ok':'bridge-warn'):'bridge-fail';
  const title=mode==='action'
    ? `SAGE produced a ${meta.label.toLowerCase()}`
    : ok
      ? 'Bridge recorded the SAGE result'
      : 'Bridge reported a SAGE failure';
  const subtitle=mode==='action'
    ? sourceLabel(info.source)
    : trusted
      ? 'Marked as trusted for finalization'
      : 'Recorded for review before finalization';
  const relations=splitRelationSummary(info.relationSummary);
  const resolved=deriveResolvedPreview(info);
  const rawValue=formatPreviewValue(info.primaryValue);
  const valuePreview=formatPreviewValue(info.valuePreview);
  const rowPreview=typeof info.rowPreview==='string'?info.rowPreview:formatRowPreview(info.rowPreview);
  const badges=[
    `<span class="atype-chip">${esc(meta.label)}</span>`,
    trusted?'<span class="bridge-badge bridge-good">Trusted final</span>':'',
    solves?'<span class="bridge-badge bridge-good">Solves task</span>':'',
    info.repairCaveat?'<span class="bridge-badge bridge-warn">Repaired path</span>':'',
    !ok?'<span class="bridge-badge bridge-bad">Failure</span>':'',
  ].filter(Boolean).join('');
  return`<div class="bridge-block ${variant}">
    <div class="bridge-top">
      <div class="bridge-head">
        <span style="font-size:16px">${esc(meta.icon)}</span>
        <div class="bridge-title-wrap">
          <div class="bridge-title">${esc(title)}</div>
          <div class="bridge-subtitle">${esc(subtitle)}</div>
        </div>
      </div>
      <div class="bridge-badges">${badges}</div>
    </div>
    <div class="bridge-primary">
      <div class="bridge-primary-k">${esc(primary.subline)}</div>
      <div class="bridge-primary-v">${esc(primary.headline)}</div>
    </div>
    <div class="bridge-body">
      ${info.semantic?`<div class="bridge-note"><span class="bridge-note-k">Interpreted as</span><span class="bridge-note-v">${esc(info.semantic)}</span></div>`:''}
      ${info.selectionBasis?`<div class="bridge-note"><span class="bridge-note-k">Why SAGE chose it</span><span class="bridge-note-v">${esc(info.selectionBasis)}</span></div>`:''}
      ${relations.length?`<div class="bridge-note"><span class="bridge-note-k">Query path</span><div class="bridge-paths">${relations.map(path=>`<div class="bridge-path">${esc(path.replace(/\s*->\s*/g,' → '))}</div>`).join('')}</div></div>`:''}
      <div class="bridge-grid">
        <div class="bridge-stat"><div class="bridge-stat-k">Result type</div><div class="bridge-stat-v">${esc(meta.label)}</div></div>
        ${info.finalVariable?`<div class="bridge-stat"><div class="bridge-stat-k">Stored as</div><div class="bridge-stat-v bridge-mono">${esc(info.finalVariable)}</div></div>`:''}
        ${info.projectedVar?`<div class="bridge-stat"><div class="bridge-stat-k">Query column</div><div class="bridge-stat-v bridge-mono">${esc(info.projectedVar)}</div></div>`:''}
        ${info.bindingCount!==''&&info.bindingCount!=null?`<div class="bridge-stat"><div class="bridge-stat-k">Raw rows</div><div class="bridge-stat-v">${esc(String(info.bindingCount))}</div></div>`:''}
        ${info.uniqueValueCount!==''&&info.uniqueValueCount!=null?`<div class="bridge-stat"><div class="bridge-stat-k">Unique values</div><div class="bridge-stat-v">${esc(String(info.uniqueValueCount))}</div></div>`:''}
        ${info.answerCardinality?`<div class="bridge-stat"><div class="bridge-stat-k">Expected shape</div><div class="bridge-stat-v">${esc(info.answerCardinality)}</div></div>`:''}
        ${info.confidence!==''&&info.confidence!=null?`<div class="bridge-stat"><div class="bridge-stat-k">Confidence</div><div class="bridge-stat-v">${esc(String(info.confidence))}</div></div>`:''}
      </div>
      ${resolved?`<div class="bridge-raw"><div class="bridge-raw-k">Resolved value</div><div class="bridge-raw-v">${esc(resolved.label)}</div><div class="bridge-raw-v bridge-mono">RAW VALUE  ${esc(resolved.raw)}</div></div>`:''}
      ${!resolved&&rawValue!=='—'?`<div class="bridge-raw"><div class="bridge-raw-k">Raw value</div><div class="bridge-raw-v bridge-mono">${esc(rawValue)}</div></div>`:''}
      ${valuePreview&&valuePreview!=='—'&&valuePreview!==rawValue?`<div class="bridge-raw"><div class="bridge-raw-k">Value preview</div><div class="bridge-raw-v bridge-mono">${esc(valuePreview)}</div></div>`:''}
      ${rowPreview?`<div class="bridge-raw"><div class="bridge-raw-k">Sample rows</div><div class="bridge-raw-v bridge-mono">${esc(rowPreview)}</div></div>`:''}
      ${info.completenessHint?`<div class="bridge-note"><span class="bridge-note-k">Coverage</span><span class="bridge-note-v">${esc(info.completenessHint)}</span></div>`:''}
      ${info.proofHint?`<div class="bridge-note"><span class="bridge-note-k">Proof hint</span><span class="bridge-note-v">${esc(info.proofHint)}</span></div>`:''}
      ${info.repairCaveat?`<div class="bridge-note"><span class="bridge-note-k">Caveat</span><span class="bridge-note-v">${esc(info.repairCaveat)}</span></div>`:''}
      ${info.failureReason?`<div class="bridge-note"><span class="bridge-note-k">Failure reason</span><span class="bridge-note-v">${esc(info.failureReason)}</span></div>`:''}
      ${info.observation?`<div class="bridge-note"><span class="bridge-note-k">Observation</span><span class="bridge-note-v">${esc(info.observation)}</span></div>`:''}
    </div>
  </div>`;
}

// ── 1. Pipeline stage timeline ─────────────────────────────────────────────
function buildStages(s){
  const turns=s.chat_history?.value||[];
  const outcome=s.evaluation_record?.outcome||'unset';
  const sage=hasBridge(s);
  const ansIcon=outcome==='correct'?'\\u2705':outcome==='incorrect'?'\\u274c':'\\u2014';
  const ansStatus=outcome==='correct'?'s-done':outcome==='incorrect'?'s-fail':'s-neutral';
  const ansStage={label:'Answer',icon:ansIcon,status:ansStatus};
  if(sage){
    const p=parseBridgePayload(turns);
    const diag=p?.sage_artifact_diagnostics||{};
    const bcount=diag.binding_count??null;
    const atype=p?.sage_artifact_type||'';
    const execOk=bcount==null||bcount>0;
    return[
      {label:'Task Received',icon:'📥',status:'s-done'},
      {label:'Query Planning',icon:'🧠',status:'s-done'},
      {label:'Code & Validate',icon:'\\u2699\\ufe0f',status:'s-done'},
      {label:'SPARQL Execution',icon:'🔍',status:execOk?'s-done':'s-warn',detail:bcount!=null?bcount+' result'+(bcount===1?'':'s'):''},
      {label:'Bridge',icon:'🌉',status:'s-done',detail:atype},
      ansStage,
    ];
  }else{
    const acts=turns.filter(t=>t.role==='agent'&&/^Action:\\s/.test((t.content||'').trim()));
    return[
      {label:'Task Received',icon:'📥',status:'s-done'},
      {label:'Tool Calls',icon:'🔧',status:acts.length?'s-done':'s-warn',detail:acts.length+' action'+(acts.length===1?'':'s')},
      ansStage,
    ];
  }
}
function renderStageTimeline(stages){
  const parts=[];
  stages.forEach((st,i)=>{
    parts.push(`<div class="stage">
      <div class="stage-dot ${esc(st.status)}">${st.icon}</div>
      <div class="stage-lbl">${esc(st.label)}</div>
      ${st.detail?`<div class="stage-det ${st.status==='s-warn'?'s-warn':''}">${esc(st.detail)}</div>`:''}
    </div>`);
    if(i<stages.length-1){
      const conn=stages.slice(0,i+1).every(x=>x.status!=='s-neutral')?'s-done':'s-pending';
      parts.push(`<div class="stage-conn ${conn}"></div>`);
    }
  });
  return`<div class="stage-wrap">${parts.join('')}</div>`;
}

// ── 3. Answer summary card ─────────────────────────────────────────────────
function renderAnswerCard(s){
  const turns=s.chat_history?.value||[];
  const outcome=s.evaluation_record?.outcome||'unset';
  const oc=ocClass(outcome);
  const f1=s.evaluation_record?.detail_dict?.f1_score;
  const f1s=f1!=null?Number(f1).toFixed(3):'N/A';
  const question=extractQuestion(turns)||'(no question found)';
  const agentRaw=s.task_output?.answer!=null?String(s.task_output.answer):'\\u2014';
  const expList=s.expected_answer?.answer_list||[];
  const expNames=s.expected_answer?.answer_name_list||[];
  // Resolve agent answer to human-readable name if possible
  let agentDisplay=agentRaw;
  if(agentRaw!=='\\u2014'){
    const parts=agentRaw.split('<SEP>');
    agentDisplay=parts.map(mid=>{const i=expList.indexOf(mid);return(i>=0&&expNames[i])?expNames[i]:mid;}).join(', ');
  }
  const expDisplay=expNames.length?expNames.join(', '):expList.join(', ')||'\\u2014';
  const expMids=expList.join(', ');
  const ocLabel=outcome==='correct'?'\\u2705  CORRECT':outcome==='incorrect'?'\\u274c  INCORRECT':'\\u2014  UNSET';
  return`<div class="ans-card oc-${esc(oc)}">
    <div class="ac-top">
      <span class="ac-outcome oc-${esc(oc)}">${ocLabel}</span>
      <span class="ac-f1">F1: ${esc(f1s)}</span>
      <span class="ac-si">Sample #${esc(s.sample_index)}</span>
    </div>
    <div class="ac-q">${esc(question)}</div>
    <div class="ac-row">
      <div class="ac-box ac-box-agent">
        <div class="ac-box-label">Agent Answer</div>
        <div class="ac-box-val">${esc(agentDisplay)}</div>
        ${agentDisplay!==agentRaw&&agentRaw!=='\\u2014'?`<div class="ac-box-mid">${esc(agentRaw)}</div>`:''}
      </div>
      <div class="ac-box ac-box-expected">
        <div class="ac-box-label">Expected Answer</div>
        <div class="ac-box-val">${esc(expDisplay)}</div>
        ${expMids&&expMids!==expDisplay?`<div class="ac-box-mid">${esc(expMids)}</div>`:''}
      </div>
    </div>
  </div>`;
}

// ── 4. SAGE execution summary (repair loop details) ─────────────────────────
function renderPalDetails(s){
  const turns=s.chat_history?.value||[];
  const p=parseBridgePayload(turns);
  if(!p)return'';
  const info=buildBridgeInfoFromPayload(p);
  const meta=artifactMeta(info.artifactType);
  const rawValue=formatPreviewValue(info.primaryValue);
  const rowPreview=Array.isArray(info.rowPreview)?formatRowPreview(info.rowPreview):String(info.rowPreview||'');
  return`<div class="sage-card">
    <div class="sage-card-title">🌉 SAGE Result Snapshot</div>
    ${info.selectionBasis?`<div style="font-size:12px;color:#d8e4f8;line-height:1.55;margin-bottom:10px">${esc(info.selectionBasis)}</div>`:''}
    <div class="sage-grid">
      <div class="sage-stat"><div class="sage-stat-k">Result Type</div><div class="sage-stat-v">${esc(meta.label)}</div></div>
      <div class="sage-stat"><div class="sage-stat-k">Query Rows</div><div class="sage-stat-v ${(Number(info.bindingCount)||0)>0?'pv-good':'pv-empty'}">${esc(String(info.bindingCount??'—'))}</div></div>
      <div class="sage-stat"><div class="sage-stat-k">Query Column</div><div class="sage-stat-v pv-mono">${esc(info.projectedVar||'—')}</div></div>
      <div class="sage-stat"><div class="sage-stat-k">Unique Values</div><div class="sage-stat-v">${esc(String(info.uniqueValueCount??'—'))}</div></div>
      <div class="sage-stat" style="grid-column:span 2"><div class="sage-stat-k">Raw Value</div><div class="sage-stat-v pv-mono" style="font-size:11px">${esc(rawValue)}</div></div>
      ${rowPreview?`<div class="sage-stat" style="grid-column:span 3"><div class="sage-stat-k">Sample Rows</div><div class="sage-stat-v pv-mono" style="font-size:11px;line-height:1.5">${esc(rowPreview)}</div></div>`:''}
    </div>
  </div>`;
}

// ── 2. Smart turn rendering ────────────────────────────────────────────────
let _obsId=0;
function renderTurnContent(t,ti){
  const c=String(t.content??'');
  const role=t.role;

  // System prompt: collapse by default
  if(ti===0&&c.startsWith(KG_PROMPT_PFX)){
    const id='p'+ti+'_'+(++_obsId);
    return`<div class="prompt-hdr" onclick="document.getElementById('${id}').classList.toggle('open')">
      📋 Task Instructions &nbsp;\\u25be (click to expand)
    </div><div class="prompt-body" id="${id}">${esc(c)}</div>`;
  }

  // Agent acknowledgment
  if(role==='agent'&&c.trim()==='OK.') return`<span class="ack">\\u2713 Agent ready</span>`;

  // Question turn
  if(role==='user'){
    const qm=c.match(/^Question:\\s*(.*?)(?=,\\s*Entities:|$)/s);
    if(qm){
      const em=c.match(/Entities:\\s*\\[([^\\]]*)\\]/);
      const ents=em?em[1].split(',').map(e=>e.trim().replace(/['"]/g,'')).filter(Boolean):[];
      return`<div class="q-block">
        <div class="q-lbl">\\u2753 Question</div>
        <div class="q-text">${esc(qm[1].trim())}</div>
        ${ents.length?`<div class="q-ents">${ents.map(e=>`<span class="ent-chip">${esc(e)}</span>`).join('')}</div>`:''}
      </div>`;
    }
  }

  // SAGE bridge macro call
  if(c.includes('sage_benchmark_bridge_macro')){
    const pm=c.match(/execute_macro\\([^,]+,\\s*(\\{[\\s\\S]*\\})\\)/);
    if(pm){
      try{return renderBridgeInfoCard(buildBridgeInfoFromPayload(JSON.parse(pm[1])),'action');}catch{}
    }
  }

  // Final Answer
  const fam=c.match(/Final\\s+Answer:\\s*#(\\d+)/i);
  if(fam) return`<div class="final-block"><span class="final-icon">\\u2705</span><span class="final-text">Final Answer: Variable #${esc(fam[1])}</span></div>`;

  // Macro result observation
  if(role==='user'&&c.includes('Macro result:')){
    const bridgeInfo=buildBridgeInfoFromMacroText(c);
    if(bridgeInfo)return renderBridgeInfoCard(bridgeInfo,'result');
    const sm=c.match(/Macro result:\\s*(\\S+)\\s*->\\s*(\\w+)/);
    const vm=c.match(/Final variable:\\s*(#\\d+)/);
    if(sm){
      const ok=sm[2]==='SUCCESS';
      return`<div class="macro-res ${ok?'macro-ok':'macro-fail'}">${ok?'\\u2713':'\\u2717'} ${esc(sm[1])} \\u2192 ${esc(sm[2])}${vm?` &nbsp;\\u00b7&nbsp; stored as <strong>${esc(vm[1])}</strong>`:''}</div>`;
    }
  }

  // Action: get_relations
  const relm=c.match(/^Action:\\s*get_relations\\((.+)\\)\\s*$/);
  if(relm) return`<div class="act-chip">🔍 <span class="act-fn">get_relations</span>(<span class="act-arg">${esc(relm[1])}</span>)</div>`;

  // Action: get_neighbors
  const nbm=c.match(/^Action:\\s*get_neighbors\\((.+?),\\s*(.+)\\)\\s*$/);
  if(nbm) return`<div class="act-chip">\\u2192 <span class="act-fn">get_neighbors</span>(<span class="act-arg">${esc(nbm[1])}</span>, <span class="act-rel">${esc(nbm[2])}</span>)</div>`;

  // Action: intersection / count / argmax / argmin
  const opm=c.match(/^Action:\\s*(intersection|count|argmax|argmin)\\((.*)\\)\\s*$/i);
  if(opm){
    const icons={intersection:'\\u2229',count:'#',argmax:'\\u2191 max',argmin:'\\u2193 min'};
    const ic=icons[opm[1].toLowerCase()]||'';
    return`<div class="act-chip"><span class="act-op">${ic}</span> <span class="act-fn">${esc(opm[1])}</span>(<span class="act-arg">${esc(opm[2])}</span>)</div>`;
  }

  // Observation / tool result (user turn matching [...] pattern)
  if(role==='user'){
    const om=c.match(/^\\[([^\\]]+)\\]\\s*([\\s\\S]*)/);
    if(om){
      const body=om[2].trim();
      const id='obs'+(++_obsId);
      const isLong=body.length>220;
      return`<div class="obs-block">
        <div class="obs-act">${esc(om[1])}</div>
        <div class="obs-body" id="${id}">${esc(isLong?body.slice(0,220):body)}</div>
        ${isLong?`<span class="obs-more" onclick="var e=document.getElementById('${id}');e.classList.toggle('expanded');this.textContent=e.classList.contains('expanded')?'show less':'show more'">show more</span>`:''}
      </div>`;
    }
  }

  // Default fallback
  return`<div class="bubble">${esc(c)}</div>`;
}

// ── 5. Sidebar: build + filter + search ───────────────────────────────────
let _filter='all';
let _search='';

function matchFilter(s){
  if(_filter==='correct') return(s.evaluation_record?.outcome||'')==='correct';
  if(_filter==='incorrect') return(s.evaluation_record?.outcome||'')==='incorrect';
  if(_filter==='sage') return hasBridge(s);
  return true;
}
function matchSearch(s){
  if(!_search)return true;
  const q=_search.toLowerCase();
  const turns=s.chat_history?.value||[];
  const question=extractQuestion(turns)||'';
  return question.toLowerCase().includes(q)||String(s.sample_index||'').includes(q);
}

function buildSidebar(){
  const correct=logData.filter(s=>(s.evaluation_record?.outcome||'')==='correct').length;
  const incorrect=logData.filter(s=>(s.evaluation_record?.outcome||'')==='incorrect').length;
  const unset=logData.length-correct-incorrect;
  document.getElementById('cnt').textContent='('+logData.length+')';
  document.getElementById('sb-stats').innerHTML=
    `<span class="stat-pill sp-c">\\u2705 ${correct}</span>`+
    `<span class="stat-pill sp-i">\\u274c ${incorrect}</span>`+
    (unset?`<span class="stat-pill sp-u">\\u2014 ${unset}</span>`:'');
  renderSidebarList();
}

function renderSidebarList(){
  const visible=logData.map((s,i)=>({s,i})).filter(({s})=>matchFilter(s)&&matchSearch(s));
  document.getElementById('sl').innerHTML=visible.map(({s,i})=>{
    const o=s.evaluation_record?.outcome||'unset';
    const sage=hasBridge(s);
    const turns=s.chat_history?.value||[];
    const q=extractQuestion(turns)||'';
    const qShort=q.length>48?q.slice(0,45)+'\\u2026':q;
    return`<div class="si" onclick="show(${i})" id="i${i}">
      <div class="si-row1">
        <span class="si-idx">Sample #${esc(s.sample_index)}</span>
        <span class="si-badges">
          ${sage?'<span class="si-sage">SAGE</span>':''}
          <span class="si-oc oc-${ocClass(o)}">${o.toUpperCase()}</span>
        </span>
      </div>
      <div class="si-q">${esc(qShort||'(no question)')}</div>
    </div>`;
  }).join('')||'<div style="padding:16px;color:#2a2a4a;font-size:12px">No matching sessions</div>';
}

// ── Main show ──────────────────────────────────────────────────────────────
function show(idx,options={}){
  _obsId=0;
  document.querySelectorAll('.si').forEach(e=>e.classList.remove('active'));
  const el=document.getElementById('i'+idx);if(el)el.classList.add('active');
  const s=logData[idx];
  const turns=s.chat_history?.value||[];
  const sage=hasBridge(s);
  const stages=buildStages(s);
  const turnsHtml=turns.map((t,ti)=>{
    const role=t.role==='agent'?'agent':'user';
    const icon=role==='agent'?'🤖':'👥';
    return`<div class="turn turn-${role}">
      <div class="t-icon">${icon}</div>
      <div class="t-body">${renderTurnContent(t,ti)}<div class="t-role">${esc(t.role)}</div></div>
    </div>`;
  }).join('')||'<div style="color:#2a2a4a;font-size:12px">No turns recorded.</div>';
  document.getElementById('main').innerHTML=
    renderAnswerCard(s)+
    renderStageTimeline(stages)+
    (sage?renderPalDetails(s):'')+
    `<div class="sec"><div class="sec-title">💬 Conversation (${turns.length} turns)</div>${turnsHtml}</div>`;
  writeState({si:String(s.sample_index??''),selectedSampleIndex:String(s.sample_index??'')});
  setHash(s.sample_index);
  if(options.restoreScroll){setTimeout(restoreScroll,80);}
  else{document.getElementById('main').scrollTop=0;persistScroll();}
}

// ── Bootstrap ──────────────────────────────────────────────────────────────
document.getElementById('sidebar').addEventListener('scroll',persistScroll,{passive:true});
document.getElementById('main').addEventListener('scroll',persistScroll,{passive:true});
window.addEventListener('beforeunload',persistScroll);
document.querySelectorAll('.flt').forEach(btn=>{
  btn.addEventListener('click',()=>{
    _filter=btn.dataset.f;
    document.querySelectorAll('.flt').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    renderSidebarList();
  });
});
document.getElementById('sb-search').addEventListener('input',e=>{
  _search=e.target.value.trim();
  renderSidebarList();
});
// ── Live-reload patch handler ───────────────────────────────────────────────
window._liveRefresh=function(newData){
  var st=readState();
  var sbEl=document.getElementById('sidebar');
  var prevSbTop=sbEl?sbEl.scrollTop:0;
  var prevSI=st.si;
  logData=window._liveData=newData;
  buildSidebar();
  if(sbEl)sbEl.scrollTop=prevSbTop;
  var curIdx=prevSI!=null?findIdx(prevSI):-1;
  if(curIdx>=0){
    document.querySelectorAll('.si').forEach(function(e){e.classList.remove('active');});
    var el=document.getElementById('i'+curIdx);
    if(el)el.classList.add('active');
  }
};
buildSidebar();
function initialIdx(){
  const hi=findIdx(readHashSI());if(hi>=0)return hi;
  const st=readState();
  const si=findIdx(st.si)>=0?findIdx(st.si):findIdx(st.selectedSampleIndex);
  return si>=0?si:0;
}
if(logData.length>0)show(initialIdx(),{restoreScroll:true});
</script></body></html>"""


def _export_html_trace(run_dir: str, session_list_data: list) -> None:
    """Write / refresh the standalone HTML trace viewer alongside runs.json.

    Kept in sync with the JSON dump — called immediately after every
    json.dump([s.model_dump() ...]) write so both files stay consistent.
    Creates run_dir/html_traces/trace_viewer.html (one file, all sessions).
    """
    html_dir = os.path.join(run_dir, "html_traces")
    os.makedirs(html_dir, exist_ok=True)
    payload = json.dumps(session_list_data, indent=2)
    html = _HTML_TRACE_TEMPLATE.replace("/*LOG_DATA_PLACEHOLDER*/", payload)
    out_path = os.path.join(html_dir, "trace_viewer.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    sessions_path = os.path.join(html_dir, "sessions.json")
    with open(sessions_path, "w", encoding="utf-8") as sfh:
        sfh.write(payload)


class ConfigUtilityCaller(StrEnum):
    CLIENT = "client"
    SERVER = "server"
    CLIENT_SIDE_CONTROLLER = "client_side_controller"


class ConfigUtility:
    def __init__(
        self,
        assignment_config: AssignmentConfig,
        environment_config: EnvironmentConfig,
        path_config: PathConfig,
    ):
        self.assignment_config = assignment_config
        self.environment_config = environment_config
        self.path_config = path_config

    def preprocess(self) -> None:
        if self.environment_config.task_client:
            self.assignment_config.task = self.environment_config.task_client

    def construct(self) -> tuple[Task[DatasetItem], Agent, dict[str, Callback]]:
        # Maybe task will be Task or TaskClient, but it doesn't matter!
        task: Task[DatasetItem] = self.assignment_config.task.create()
        # region Construct language_model_dict
        # Here, We actually instantiate the language models.
        # After the code exit construct(), We will never get a chance to get the language model instance.
        # But I think this is good, since this improve the modularity and maintainability of the code.
        language_model_dict: Mapping[str, LanguageModel] = {
            key: value.create()
            for key, value in self.assignment_config.language_model_dict.items()
        }
        agent_instance_factory: GeneralInstanceFactory = self.assignment_config.agent
        if (
            language_model_name := agent_instance_factory.parameters.get(
                "language_model"
            )
        ) is not None:
            agent_instance_factory.parameters["language_model"] = language_model_dict[
                language_model_name
            ]
        # endregion
        agent: Agent = agent_instance_factory.create()
        callback_dict = CallbackConstructor.construct(
            self.assignment_config, task, agent, language_model_dict
        )
        return task, agent, callback_dict

    def validate(self, task: Task[DatasetItem], agent: Agent) -> None:
        sample_index_list = task.get_sample_index_list()
        for selected_sample_index in self.assignment_config.sample_order:
            assert selected_sample_index in sample_index_list

    def postprocess(self, task: Task[DatasetItem], agent: Agent) -> None:
        if self.assignment_config.sample_order == "default":
            self.assignment_config.sample_order = task.get_sample_index_list()

    def remove_redundant_args(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        # Maybe use `if raw_config["environment_config"]["use_task_client_flag"]` is better, but I use the following
        # condition avoid using dict key directly.
        if not self.environment_config.task_client:
            # If the config file is used to restore the previous incomplete assignment, the `task_client` will be None.
            # Using del to remove the key-value pair will cause an error in this case.
            for key in list(raw_config["environment_config"]):
                if key != "use_task_client_flag":
                    del raw_config["environment_config"][key]
        redundant_key_buffer: set[tuple[str, str]] = set()
        for key in raw_config["task_dict"]:
            if key != raw_config["assignment_config"]["task"]:
                redundant_key_buffer.add(("task_dict", key))
        for key in raw_config["agent_dict"]:
            if key != raw_config["assignment_config"]["agent"]["name"]:
                redundant_key_buffer.add(("agent_dict", key))
        assignment_language_model_name_list: Sequence[str] = [
            language_model_info_dict["name"]
            for language_model_info_dict in raw_config["assignment_config"][
                "language_model_list"
            ]
        ]
        for key in raw_config["language_model_dict"]:
            if key not in assignment_language_model_name_list:
                redundant_key_buffer.add(("language_model_dict", key))
        assignment_callback_name_list: Sequence[str] = [
            callback_info_dict["name"]
            for callback_info_dict in raw_config["assignment_config"][
                "callback_dict"
            ].values()
        ]
        for key in raw_config["callback_dict"]:
            if key not in assignment_callback_name_list:
                redundant_key_buffer.add(("callback_dict", key))
        for info_tuple in redundant_key_buffer:
            del raw_config[info_tuple[0]][info_tuple[1]]
        return raw_config

    @staticmethod
    def _get_custom_instance_info_dict(
        default_instance_info_dict: Mapping[str, Any],
        custom_instance_info_dict: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        module: str = default_instance_info_dict["module"]
        # `... or {}` is used to ensure that both default_parameters and custom_parameters are dict.
        default_parameters: dict[str, Any] = copy.deepcopy(
            default_instance_info_dict.get("parameters") or {}
        )
        custom_parameters = custom_instance_info_dict.get("custom_parameters") or {}
        for parameter_name, custom_parameter_value in custom_parameters.items():
            assert parameter_name in default_parameters  # Do not remove this assertion.
            # Overwrite the default parameter value with the custom parameter value.
            default_parameters[parameter_name] = custom_parameter_value
        return {
            "module": module,
            "parameters": default_parameters,
        }

    @staticmethod
    def read_raw_config(
        raw_config: Mapping[str, Any], caller: ConfigUtilityCaller
    ) -> tuple[AssignmentConfig, EnvironmentConfig, LoggerConfig, PathConfig]:
        raw_config = copy.deepcopy(raw_config)  # Avoid modifying the original config.
        output_dir_override = get_output_dir_override()
        if output_dir_override:
            raw_config["assignment_config"]["output_dir"] = output_dir_override
        # region Convert raw_config into assignment_config
        # region Construct assignment_language_model_dict
        assignment_language_model_list: Sequence[Mapping[str, Any]] = raw_config[
            "assignment_config"
        ]["language_model_list"]
        assignment_language_model_dict: dict[str, Any] = {}
        for language_model_info_dict in assignment_language_model_list:
            language_model_name = language_model_info_dict["name"]
            default_language_model_info_dict = raw_config["language_model_dict"][
                language_model_name
            ]
            assert language_model_name not in assignment_language_model_dict
            assignment_language_model_dict[language_model_name] = (
                ConfigUtility._get_custom_instance_info_dict(
                    default_language_model_info_dict, language_model_info_dict
                )
            )
        # endregion
        # region Construct assignment_agent
        custom_agent_info_dict = raw_config["assignment_config"]["agent"]
        default_agent_info_dict = raw_config["agent_dict"][
            custom_agent_info_dict["name"]
        ]
        assignment_agent_info_dict = ConfigUtility._get_custom_instance_info_dict(
            default_agent_info_dict, custom_agent_info_dict
        )
        assignment_agent = GeneralInstanceFactory.model_validate(
            assignment_agent_info_dict
        )
        if (
            language_model_name := assignment_agent.parameters.get("language_model")
        ) is not None:
            # Do not replace the language_model in the parameters with the GeneralInstanceFactory instance.
            if language_model_name not in assignment_language_model_dict:
                default_language_model_info_dict = raw_config["language_model_dict"].get(
                    language_model_name
                )
                if default_language_model_info_dict is None:
                    raise ValueError(
                        f"Unknown language_model {language_model_name!r} in assignment agent config"
                    )
                assignment_language_model_dict[language_model_name] = (
                    ConfigUtility._get_custom_instance_info_dict(
                        default_language_model_info_dict, {"name": language_model_name}
                    )
                )
        # endregion
        # region Construct assignment_callback_dict
        assignment_callback_dict: dict[str, Any] = raw_config["assignment_config"][
            "callback_dict"
        ]
        generated_tool_callback_name = "generated_tool_logging_callback"
        if generated_tool_callback_name in raw_config.get("callback_dict", {}):
            already_enabled = any(
                info.get("name") == generated_tool_callback_name
                for info in assignment_callback_dict.values()
            )
            if not already_enabled:
                assignment_callback_dict["callback_generated_tool_logging"] = {
                    "name": generated_tool_callback_name
                }
        for callback_key, callback_info_dict in assignment_callback_dict.items():
            default_callback_info_dict = raw_config["callback_dict"][
                callback_info_dict["name"]
            ]
            assignment_callback_dict[callback_key] = (
                ConfigUtility._get_custom_instance_info_dict(
                    default_callback_info_dict, callback_info_dict
                )
            )
        # endregion
        assignment_config = AssignmentConfig(
            task=raw_config["task_dict"][raw_config["assignment_config"]["task"]],
            agent=assignment_agent,
            language_model_dict=assignment_language_model_dict,
            output_dir=raw_config["assignment_config"]["output_dir"],
            sample_order=raw_config["assignment_config"]["sample_order"],
            callback_dict=assignment_callback_dict,
        )
        # endregion
        # region Convert raw_config into environment_config
        if raw_config["environment_config"]["use_task_client_flag"]:
            environment_config = EnvironmentConfig(
                task_client=raw_config["environment_config"]["task_client"],
                chat_history_item_factory_client=raw_config["environment_config"][
                    "chat_history_item_factory_client"
                ],
                server_side_controller_address=raw_config["environment_config"][
                    "server_side_controller_address"
                ],
                interpreter_path=raw_config["environment_config"]["interpreter_path"],
            )
        else:
            environment_config = EnvironmentConfig(
                task_client=None,
                chat_history_item_factory_client=None,
                server_side_controller_address=None,
                interpreter_path=None,
            )
        # endregion
        # region Convert raw_config into logger_config
        if raw_config["logger_config"]["log_file_path"] == "default":
            if raw_config["environment_config"]["use_task_client_flag"]:
                match caller:
                    case ConfigUtilityCaller.CLIENT:
                        log_file_path = os.path.join(
                            assignment_config.output_dir,
                            prefix_filename("singleton_logger_client.log"),
                        )
                    case ConfigUtilityCaller.SERVER:
                        log_file_path = os.path.join(
                            assignment_config.output_dir,
                            prefix_filename("singleton_logger_server.log"),
                        )
                    case ConfigUtilityCaller.CLIENT_SIDE_CONTROLLER:
                        log_file_path = (
                            "./outputs/singleton_logger_client_side_controller.log"
                        )
                    case _:
                        raise NotImplementedError()
            else:
                log_file_path = os.path.join(
                    assignment_config.output_dir,
                    prefix_filename("singleton_logger.log"),
                )
        else:
            log_file_path = raw_config["logger_config"]["log_file_path"]
        logger_config = LoggerConfig(
            level=raw_config["logger_config"]["level"],
            log_file_path=log_file_path,
            logger_name=raw_config["logger_config"]["logger_name"],
        )
        # endregion
        # region Construct path_config from assignment_config
        path_config = PathConfig(
            exception_record_file_path=os.path.join(
                assignment_config.output_dir, prefix_filename("exception.txt")
            ),
            config_output_path=os.path.join(
                assignment_config.output_dir, prefix_filename("config.yaml")
            ),
            session_list_output_path=os.path.join(
                assignment_config.output_dir, prefix_filename("runs.json")
            ),
            metric_output_path=os.path.join(
                assignment_config.output_dir, prefix_filename("metric.json")
            ),
            coredumpy_output_dir=os.path.join(
                assignment_config.output_dir, prefix_filename("coredumpy")
            ),
        )
        # endregion
        return assignment_config, environment_config, logger_config, path_config

    @staticmethod
    def is_raw_config_equal(
        raw_config_1: dict[str, Any], raw_config_2: dict[str, Any]
    ) -> bool:
        raw_config_1 = copy.deepcopy(raw_config_1)
        raw_config_2 = copy.deepcopy(raw_config_2)
        output_dir_1 = raw_config_1["assignment_config"].pop("output_dir")
        output_dir_2 = raw_config_2["assignment_config"].pop("output_dir")
        return raw_config_1 == raw_config_2 and AssignmentConfig.is_output_dir_equal(
            output_dir_1, output_dir_2
        )


def _maybe_enable_sage_agent(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    if not ENABLE_SAGE_AGENT:
        return copy.deepcopy(raw_config)

    updated_raw_config = copy.deepcopy(raw_config)
    sage_agent_component = ConfigLoader().load_from(SAGE_AGENT_COMPONENT_CONFIG_PATH)
    updated_raw_config.setdefault("agent_dict", {})
    updated_raw_config["agent_dict"].update(sage_agent_component)

    current_agent_info = updated_raw_config["assignment_config"].get("agent") or {}
    current_custom_parameters = current_agent_info.get("custom_parameters") or {}
    language_model_name = current_custom_parameters.get("language_model")
    if language_model_name is None:
        assignment_language_model_list = (
            updated_raw_config["assignment_config"].get("language_model_list") or []
        )
        if assignment_language_model_list:
            language_model_name = assignment_language_model_list[0]["name"]
    if language_model_name is None:
        raise ValueError("ENABLE_SAGE_AGENT=1 requires an assignment language model.")

    inference_config_override: dict[str, Any] = {}
    raw_max_completion_tokens = os.environ.get(
        SAGE_AGENT_MAX_COMPLETION_TOKENS_ENV, ""
    ).strip()
    if raw_max_completion_tokens:
        inference_config_override["max_completion_tokens"] = int(
            raw_max_completion_tokens
        )
    raw_reasoning_effort = os.environ.get(
        SAGE_AGENT_REASONING_EFFORT_ENV, ""
    ).strip()
    if raw_reasoning_effort:
        inference_config_override["reasoning_effort"] = raw_reasoning_effort
    updated_raw_config["assignment_config"]["agent"] = {
        "name": SAGE_AGENT_NAME,
        "custom_parameters": {
            "language_model": language_model_name,
            "inference_config_dict": inference_config_override,
        },
    }
    return updated_raw_config


def _get_language_model_override_name() -> Optional[str]:
    for env_name in (
        LANGUAGE_MODEL_OVERRIDE_ENV,
        LANGUAGE_MODEL_OVERRIDE_FALLBACK_ENV,
    ):
        raw_value = str(os.environ.get(env_name) or "").strip()
        if raw_value:
            return raw_value
    return None


def _maybe_override_language_model(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    override_name = _get_language_model_override_name()
    if not override_name:
        return copy.deepcopy(raw_config)

    updated_raw_config = copy.deepcopy(raw_config)
    language_model_dict = updated_raw_config.get("language_model_dict") or {}
    if override_name not in language_model_dict:
        raise ValueError(
            "Language model override "
            f"'{override_name}' not found in language_model_dict."
        )

    assignment_config = updated_raw_config.setdefault("assignment_config", {})
    assignment_config["language_model_list"] = [{"name": override_name}]

    current_agent_info = assignment_config.get("agent") or {}
    if current_agent_info:
        current_agent_info = copy.deepcopy(current_agent_info)
        current_custom_parameters = copy.deepcopy(
            current_agent_info.get("custom_parameters") or {}
        )
        current_custom_parameters["language_model"] = override_name
        current_agent_info["custom_parameters"] = current_custom_parameters
        assignment_config["agent"] = current_agent_info

    return updated_raw_config


def main() -> None:
    # region Prepare variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()
    raw_config = ConfigLoader().load_from(args.config_path)
    raw_config = _maybe_override_language_model(raw_config)
    raw_config = _maybe_enable_sage_agent(raw_config)
    assignment_config, environment_config, logger_config, path_config = (
        ConfigUtility.read_raw_config(raw_config, ConfigUtilityCaller.CLIENT)
    )
    # Prepare the variable that will be used in the following procedures.
    config_utility = ConfigUtility(assignment_config, environment_config, path_config)
    # endregion
    # region write raw_config to disk
    cleaned_config = config_utility.remove_redundant_args(raw_config)
    config_output_path = path_config.config_output_path
    if os.path.exists(config_output_path):
        config_from_disk = yaml.safe_load(open(config_output_path, "r"))
        assert ConfigUtility.is_raw_config_equal(config_from_disk, cleaned_config)
        # The config file already exists, so we don't need to write it again.
    else:
        # Write the config file to the output directory.
        config_output_dir = os.path.dirname(config_output_path)
        os.makedirs(config_output_dir, exist_ok=True)
        yaml.dump(
            cleaned_config,
            open(config_output_path, "w"),
        )
    # endregion
    # region Initialize logger, Set coredumpy output dir
    logger = SingletonLogger.get_instance(logger_config)
    coredumpy.patch_except(directory=path_config.coredumpy_output_dir)
    # endregion
    # region Configure model output logging path
    os.environ.setdefault(
        "LIFELONG_MODEL_OUTPUT_PATH",
        os.path.join(
            assignment_config.output_dir, prefix_filename("model_outputs.jsonl")
        ),
    )
    # endregion
    # region Construct variable, valid config
    config_utility.preprocess()
    task, agent, callback_dict = config_utility.construct()
    # Late-bind: give the agent a reference to the KG task so Macro tools
    # can be routed to the server-side _execute_macro engine.
    # IMPORTANT: Do NOT use hasattr(task, ...) here — if `task` is a
    # Client proxy, hasattr triggers Client.__getattr__ which makes a
    # network call to /get_attribute before the server is ready.
    if hasattr(agent, "_route_macro_to_server"):
        agent._kg_task_ref = task
    config_utility.postprocess(task, agent)
    config_utility.validate(task, agent)
    ContinualAgentBenchException.set_record_file(path_config.exception_record_file_path)
    # endregion
    # region Determine whether to start a new assignment or restore the previous incomplete assignment, based on
    # whether the config file exists.
    session_list_output_path = path_config.session_list_output_path
    assert isinstance(assignment_config.sample_order, list)
    session_list: list[Session]
    unfinished_sample_order: list[SampleIndex]
    if os.path.exists(session_list_output_path):
        # At least one session exists, so we restore the previous incomplete assignment.
        session_list = [
            Session.model_validate(session_info_dict)
            for session_info_dict in json.load(open(session_list_output_path, "r"))
        ]
        unfinished_sample_order = [
            sample_index
            for sample_index in assignment_config.sample_order
            if all(session.sample_index != sample_index for session in session_list)
        ]
        # Previous session may change the state of the callback, restore it here.
        CallbackRestorer.restore(callback_dict)
    else:
        # Start a new assignment.
        session_list = []
        unfinished_sample_order = assignment_config.sample_order
    callback_handler = CallbackHandler(callback_dict)
    # endregion
    # region Run experiment
    logger.info(
        f"Experiment start. "
        f"Total sample count: {len(assignment_config.sample_order)}. "
        f"Unfinished sample count: {len(unfinished_sample_order)}."
    )
    for sample_index in unfinished_sample_order:
        # region Initialize session
        session = Session(task_name=task.task_name, sample_index=sample_index)
        callback_args = CallbackArguments(
            current_session=session, task=task, agent=agent, session_list=session_list
        )
        callback_handler.on_session_create(callback_args)
        if callback_args.session_controller.should_task_reset:
            task.reset(session)
            callback_handler.on_task_reset(callback_args)
        logger.info(f"Sample {sample_index} start.")
        # endregion
        # region Run session
        while session.sample_status == SampleStatus.RUNNING:
            if callback_args.session_controller.should_agent_inference:
                agent._current_session = session
                agent.inference(session)
                callback_handler.on_agent_inference(callback_args)
            if callback_args.session_controller.should_task_interact:
                task.interact(session)
                callback_handler.on_task_interact(callback_args)
        # endregion
        # region Complete session
        if callback_args.session_controller.should_task_complete:
            task.complete(session)
            callback_handler.on_task_complete(callback_args)
        # Reflection Forge: if the task failed, trigger strict post-task
        # tool generation so the agent can learn from its mistakes.
        # Gated by ENABLE_POST_TASK_REFLECTION (default False).
        # The mid-run Orchestrator escape-hatch (request_new_tool) is
        # unaffected by this gate.
        if ENABLE_POST_TASK_REFLECTION:
            try:
                is_incorrect = (
                    session.evaluation_record.outcome == SessionEvaluationOutcome.INCORRECT
                )
                is_limit_reached = (
                    session.finish_reason is not None
                    and "task_limit" in session.finish_reason
                )
                if (is_incorrect or is_limit_reached) and hasattr(
                    agent, "generate_tool_from_failure"
                ):
                    agent.generate_tool_from_failure(
                        session,
                        expected_answer=session.expected_answer,
                    )
                    logger.info(
                        f"Sample {sample_index}: Reflection Forge completed."
                    )
            except Exception:
                logger.exception(
                    f"Sample {sample_index}: Reflection Forge failed."
                )
        # Dynamic quality feedback: reward/penalize invoked tools.
        try:
            is_correct = (
                session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
            )
            if hasattr(agent, "update_tool_quality_for_outcome"):
                agent.update_tool_quality_for_outcome(success=is_correct)
                logger.info(
                    f"Sample {sample_index}: Tool quality updated "
                    f"(success={is_correct})."
                )
        except Exception:
            logger.exception(
                f"Sample {sample_index}: Tool quality update failed."
            )
        session_list.append(session)
        _session_dumps = [s.model_dump() for s in session_list]
        json.dump(
            _session_dumps,
            open(session_list_output_path, "w"),  # noqa
            indent=2,
        )
        try:
            _export_html_trace(
                os.path.dirname(session_list_output_path), _session_dumps
            )
        except Exception:
            pass  # never block the main experiment loop
        logger.info(
            f"Sample {sample_index} end. Session status: {session.sample_status}. "
            f"Evaluation outcome: {session.evaluation_record.outcome}."
        )
        # endregion
        # region Save callback state
        # The state of callback will be used to restore the previous incomplete assignment.
        callback_handler.on_state_save(callback_args)
        # endregion
    # endregion
    # region Evaluate
    session_metric_calculation_partial_list: Sequence[
        SessionMetricCalculationPartial
    ] = [
        SessionMetricCalculationPartial(
            sample_index=session.sample_index,
            evaluation_record=session.evaluation_record,
            sample_status=session.sample_status,
        )
        for session in session_list
    ]
    metric = task.calculate_metric(session_metric_calculation_partial_list)
    logger.info(
        f"Experiment end. Metric: {metric}. Total sample count: {len(assignment_config.sample_order)}.",
    )
    json.dump(
        metric,
        open(path_config.metric_output_path, "w"),  # noqa
        indent=2,
    )
    logger.info(f"Metric file has been saved to {assignment_config.output_dir}.")
    # Write a simple per-task outcome summary for quick review.
    completed_count = 0
    not_completed_count = 0
    outcome_rows = []
    for session in session_list:
        completed = session.sample_status == SampleStatus.COMPLETED
        if completed:
            completed_count += 1
        else:
            not_completed_count += 1
        outcome_rows.append(
            {
                "sample_index": session.sample_index,
                "task_name": str(session.task_name),
                "completed": completed,
                "correct": session.evaluation_record.outcome
                == SessionEvaluationOutcome.CORRECT,
                "outcome": str(session.evaluation_record.outcome),
                "not_completed_reason": None
                if completed
                else (session.finish_reason or str(session.sample_status)),
            }
        )
    task_summary = {
        "total_samples": len(session_list),
        "completed_count": completed_count,
        "not_completed_count": not_completed_count,
        "results": outcome_rows,
    }
    summary_path = os.path.join(
        assignment_config.output_dir, prefix_filename("task_outcomes.json")
    )
    json.dump(task_summary, open(summary_path, "w"), indent=2)  # noqa
    logger.info("Task outcome summary saved to %s.", summary_path)
    # endregion
    # region Release
    task.release()
    # endregion
    print(
        "\n" + "=" * 70
        + "\n[CLEANUP REMINDER] Before running the next benchmark, manually delete"
        "\n any existing 'Macro' tools in your generated_tools directory that"
        "\n were created before these anti-contamination changes. Old tools may"
        "\n carry hardcoded domain entities (semantic contamination)."
        "\n" + "=" * 70
    )


if __name__ == "__main__":
    main()

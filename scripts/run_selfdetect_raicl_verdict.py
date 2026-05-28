#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, random, re, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = PROJECT_ROOT / "refactor" / "pre_atom_pipeline" / "output" / "selfdetect_raicl_verdict"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src" / "step9_self_correction" / "v2"))
from note_span_index import topk_spans  # noqa: E402

DET_SYSTEM = "You are a careful clinical QA auditor. You must check whether an answer is supported by the discharge note."
DET_P5 = """Discharge note:
{note}

Question:
{question}

Answer to audit:
{answer}

Your job is not only to decide whether the answer is wrong. Your job is to create a correction payload that a downstream retrieval step can use.

Check in this order:
1. QUESTION_FOCUS: What exact visit/date/aspect/fact does the question ask for?
2. ANSWER_FOCUS: What does the answer actually focus on?
3. WRONG_OR_MISSING_TARGET: If wrong, identify the smallest wrong claim or missing required fact.
4. EVIDENCE_NEEDED: What note evidence would prove the correction?

Only mark INCORRECT if the issue changes the answer to the question. Do not flag minor extra details.

Return exactly this template:
VERDICT: CORRECT or INCORRECT
ERROR_TYPE: CONTRADICTION or OMISSION or QUESTION_MISALIGNMENT or NONE
QUESTION_FOCUS: one sentence
ANSWER_FOCUS: one sentence
WRONG_CLAIM: the smallest wrong claim, or NONE
CORRECT_OR_MISSING_INFO: the fact that should replace/add/refocus the answer, or NONE
EVIDENCE_NEEDED: what kind of note span should be retrieved
RETRIEVAL_QUERY_1: short query using the question focus
RETRIEVAL_QUERY_2: short query using the wrong/missing target
RETRIEVAL_QUERY_3: short query using key clinical entities
CORRECTION_HINT: one sentence telling the downstream corrector what to change
WHY: one short explanation"""

PARSE_SYSTEM = "Extract structured fields from a clinical self-audit. Return JSON only."
PARSE_DET_USER = """Extract the detection payload from this text. Use only what the text says; do not re-judge the clinical case.

TEXT:
{raw}

Return JSON:
{{"verdict":"CORRECT|INCORRECT|UNCLEAR", "error_type":"CONTRADICTION|OMISSION|QUESTION_MISALIGNMENT|NONE|UNCLEAR", "question_focus":"string", "answer_focus":"string", "wrong_claim":"string", "correct_or_missing_info":"string", "evidence_needed":"string", "retrieval_queries":["string"], "correction_hint":"string", "why":"string"}}"""

COR_SYSTEM = "You are a careful clinical QA assistant. Revise only when same-patient evidence supports the revision."
COR_USER = """Discharge note:
{note}

Question:
{question}

Previous answer:
{answer}

Detection feedback:
- error type: {error_type}
- question focus: {question_focus}
- wrong claim: {wrong_claim}
- correction target: {correct_or_missing_info}
- correction hint: {correction_hint}

Same-patient retrieved evidence. This is the factual source:
{spans_block}

Retrieved correction example from another patient. Use as style/pattern only, not factual content:
{example_block}

Write the best final answer to the question in 1-3 sentences. Use only facts supported by the discharge note and evidence spans."""

VERDICT_SYSTEM = "You are a medical expert comparing two answers to the same clinical question."
VERDICT_USER = """Discharge summary:
{note}

Question:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Pick the answer that better answers the question using the discharge summary.
Consider: consistency with notes, completeness for all parts of the question, and directness.
If the corrected answer introduces unsupported facts or drops necessary information, prefer the original.

Reply on the FIRST line with exactly one letter: A or B
On the SECOND line, give one short reason."""
PARSE_VERDICT_USER = """The text below was supposed to pick answer A or B. Extract the pick. If unclear, return UNCLEAR.

TEXT:
{raw}

Return JSON: {{"pick":"A|B|UNCLEAR", "reason":"string"}}"""

JUDGE_SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."

def judge_user(note: str, question: str, ground_truth: str, answer: str) -> str:
    return f"DISCHARGE SUMMARY:\n{note}\n\nQUESTION:\n{question}\n\nCORRECT ANSWER (Ground Truth):\n{ground_truth}\n\nMODEL'S ANSWER:\n{answer}\n\nTask: Evaluate if the model's answer is correct compared to the ground truth.\n\nRespond with ONLY a single digit:\n1 = Correct\n0 = Incorrect"


def load_api_key() -> str:
    env=PROJECT_ROOT/'.env'
    if env.exists():
        for line in env.read_text().splitlines():
            line=line.strip()
            if line.startswith('OPENAI_API_KEY=') and not line.startswith('#'):
                return line.split('=',1)[1].strip()
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    raise RuntimeError('OPENAI_API_KEY not found')

_client: OpenAI|None=None
def openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client=OpenAI(api_key=load_api_key())
    return _client


def served_model_id(port:int)->str:
    r=requests.get(f'http://localhost:{port}/v1/models',timeout=10); r.raise_for_status(); return r.json()['data'][0]['id']

def strip_think(text:str)->str:
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S|re.I).strip()
    if '</think>' in text.lower(): text=re.sub(r'^.*?</think>\s*','',text,flags=re.S|re.I).strip()
    return text

def vllm_chat(system:str,user:str,port:int,max_tokens:int,temperature:float)->str:
    model=served_model_id(port)
    payload={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':user}], 'max_tokens':max_tokens,'temperature':temperature}
    r=requests.post(f'http://localhost:{port}/v1/chat/completions',json=payload,timeout=300)
    body=r.json()
    if 'choices' not in body:
        payload['messages']=[{'role':'user','content':f'{system}\n\n{user}'}]
        r=requests.post(f'http://localhost:{port}/v1/chat/completions',json=payload,timeout=300); body=r.json()
    if 'choices' not in body: raise RuntimeError(str(body))
    return strip_think((body['choices'][0]['message']['content'] or '').strip())


def parse_binary(text:str|None)->int|None:
    if text is None: return None
    if '1' in text and '0' not in text: return 1
    if '0' in text: return 0
    return None


def gpt_json(user:str, max_tokens:int=300)->dict[str,Any]:
    for attempt in range(4):
        try:
            r=openai_client().chat.completions.create(model='gpt-4o-mini',messages=[{'role':'system','content':PARSE_SYSTEM},{'role':'user','content':user}],temperature=0.0,max_tokens=max_tokens,response_format={'type':'json_object'})
            return json.loads((r.choices[0].message.content or '{}').strip())
        except Exception as e:
            if attempt==3: return {'error':str(e)}
            time.sleep(2*(attempt+1))
    return {}


def gpt_judge(note:str,question:str,gt:str,answer:str)->dict[str,Any]:
    for attempt in range(5):
        try:
            r=openai_client().chat.completions.create(model='gpt-4o',messages=[{'role':'system','content':JUDGE_SYSTEM},{'role':'user','content':judge_user(note,question,gt,answer)}],temperature=0.1,max_tokens=10)
            raw=(r.choices[0].message.content or '').strip(); return {'label':parse_binary(raw),'raw':raw,'temperature':0.1,'model':'gpt-4o'}
        except Exception as e:
            if attempt==4: return {'label':None,'raw':'','error':str(e)}
            time.sleep(2*(attempt+1))
    return {'label':None}


def field(text:str,name:str)->str:
    m=re.search(rf'^\s*\*?\*?{re.escape(name)}\*?\*?\s*:\s*(.+)$',text or '',re.I|re.M)
    return m.group(1).strip() if m else ''

def parse_detection_regex(raw:str)->dict[str,Any]:
    verdict_s=field(raw,'VERDICT').upper()
    verdict='INCORRECT' if 'INCORRECT' in verdict_s else ('CORRECT' if 'CORRECT' in verdict_s else 'UNCLEAR')
    et_s=field(raw,'ERROR_TYPE').upper(); et='UNCLEAR'
    for t in ['QUESTION_MISALIGNMENT','CONTRADICTION','OMISSION','NONE']:
        if t in et_s: et=t; break
    qs=[field(raw,f'RETRIEVAL_QUERY_{i}') for i in (1,2,3)]
    return {'verdict':verdict,'error_type':et,'question_focus':field(raw,'QUESTION_FOCUS'),'answer_focus':field(raw,'ANSWER_FOCUS'),'wrong_claim':field(raw,'WRONG_CLAIM'),'correct_or_missing_info':field(raw,'CORRECT_OR_MISSING_INFO'),'evidence_needed':field(raw,'EVIDENCE_NEEDED'),'retrieval_queries':[q for q in qs if q and q.upper()!='NONE'],'correction_hint':field(raw,'CORRECTION_HINT'),'why':field(raw,'WHY'),'parse_path':'regex'}


def valid_detection(p:dict[str,Any])->bool:
    if p.get('verdict')=='CORRECT': return True
    if p.get('verdict')!='INCORRECT': return False
    if p.get('error_type') not in {'CONTRADICTION','OMISSION','QUESTION_MISALIGNMENT'}: return False
    if not (p.get('wrong_claim') or p.get('correct_or_missing_info')): return False
    if not (p.get('retrieval_queries') or p.get('evidence_needed') or p.get('question_focus')): return False
    return True


def parse_detection(raw:str)->dict[str,Any]:
    p=parse_detection_regex(raw)
    if valid_detection(p): return p
    obj=gpt_json(PARSE_DET_USER.format(raw=(raw or '')[:5000]))
    out={'verdict':str(obj.get('verdict','UNCLEAR')).upper(),'error_type':str(obj.get('error_type','UNCLEAR')).upper(),'question_focus':str(obj.get('question_focus','')),'answer_focus':str(obj.get('answer_focus','')),'wrong_claim':str(obj.get('wrong_claim','')),'correct_or_missing_info':str(obj.get('correct_or_missing_info','')),'evidence_needed':str(obj.get('evidence_needed','')),'retrieval_queries':obj.get('retrieval_queries',[]) if isinstance(obj.get('retrieval_queries',[]),list) else [],'correction_hint':str(obj.get('correction_hint','')),'why':str(obj.get('why','')),'parse_path':'gpt4o-mini'}
    if valid_detection(out): return out
    out['verdict']='UNCLEAR'; out['valid']=False
    return out


def parse_verdict(raw:str)->dict[str,Any]:
    first=(raw or '').strip().splitlines()[0] if (raw or '').strip() else ''
    m=re.match(r'^[\s\-*#>]*([AaBb])\b',first)
    if m: return {'pick':m.group(1).upper(),'parse_path':'regex','reason':' '.join((raw or '').splitlines()[1:])[:200]}
    obj=gpt_json(PARSE_VERDICT_USER.format(raw=(raw or '')[:2000]),max_tokens=120)
    pick=str(obj.get('pick','UNCLEAR')).upper()
    if pick not in {'A','B'}: pick='UNCLEAR'
    return {'pick':pick,'parse_path':'gpt4o-mini','reason':str(obj.get('reason',''))[:200]}


def load_notes()->dict[str,str]:
    df=pd.read_json(PROJECT_ROOT/'output'/'EHRNoteQA_processed.jsonl',lines=True); out={}
    for _,r in df.iterrows():
        parts=[]
        for i in (1,2,3):
            t=r.get(f'note_{i}')
            if pd.notna(t) and str(t).strip() and str(t).strip().lower()!='nan': parts.append(f'[Note {i}]\n{str(t).strip()}')
        out[str(int(r['patient_id']))]='\n\n'.join(parts)
    return out


def load_rows(n_wrong:int,n_correct:int,seed:int)->list[dict[str,Any]]:
    notes=load_notes(); rows=[]
    for fold in range(5):
        df=pd.read_csv(PROJECT_ROOT/'output'/'step8'/'qwen2.5-7b-instruct'/f'fold_{fold}'/'zeroshot_evaluated_binary.csv')
        for _,r in df.iterrows():
            pid=int(r['patient_id']); rows.append({'fold':fold,'idx':int(r['idx']),'patient_id':pid,'question':str(r['question']),'ground_truth':str(r['ground_truth']),'answer':str(r['model_answer']),'orig_label':int(r['binary_correct']),'note':notes[str(pid)]})
    wrong=[r for r in rows if r['orig_label']==0]; correct=[r for r in rows if r['orig_label']==1]
    rng=random.Random(seed); rng.shuffle(wrong); rng.shuffle(correct)
    if n_wrong<0: n_wrong=len(wrong)
    if n_correct<0: n_correct=len(correct)
    sample=wrong[:min(n_wrong,len(wrong))]+correct[:min(n_correct,len(correct))]; rng.shuffle(sample); return sample

_pool_cache={}
def load_pool(fold:int)->list[dict[str,Any]]:
    if fold not in _pool_cache:
        p=PROJECT_ROOT/'workspace'/'self_critique'/'data'/'bm_contrast_pool'/f'fold_{fold}_pool.json'
        _pool_cache[fold]=json.loads(p.read_text()) if p.exists() else []
    return _pool_cache[fold]

def toks(s:str)->set[str]: return set(re.findall(r'[a-zA-Z0-9]+',(s or '').lower()))
def retrieve_example(row:dict[str,Any], det:dict[str,Any])->dict[str,Any]|None:
    pool=load_pool(row['fold']);
    if not pool: return None
    query=' '.join([row['question'],det.get('error_type',''),det.get('question_focus',''),det.get('wrong_claim',''),det.get('correct_or_missing_info','')])
    qt=toks(query)
    def score(ex):
        text=' '.join([ex.get('question',''),ex.get('what_was_wrong',''),ex.get('ground_truth','')])
        return len(qt & toks(text))
    return max(pool,key=score)

def retrieve_spans(row:dict[str,Any], det:dict[str,Any], k:int)->list[dict[str,Any]]:
    queries=[row['question'],det.get('question_focus',''),det.get('wrong_claim',''),det.get('correct_or_missing_info',''),det.get('evidence_needed','')]+list(det.get('retrieval_queries') or [])
    return topk_spans(row['note'],queries,k=k,scoring='agreement')

def render_spans(spans:list[dict[str,Any]])->str:
    return '\n'.join(f'[{i+1}] {s["sentence"]}' for i,s in enumerate(spans)) if spans else '(none)'

def render_example(ex:dict[str,Any]|None)->str:
    if not ex: return '(none)'
    ev='; '.join(ex.get('evidence_from_notes') or [])[:600]
    return f"Question: {ex.get('question','')}\nWrong answer: {ex.get('wrong_answer','')}\nWhat was wrong: {ex.get('what_was_wrong','')}\nCorrect answer pattern: {ex.get('ground_truth','')}\nEvidence style: {ev}"


def run_detect(row,port,temp)->dict[str,Any]:
    raw=vllm_chat(DET_SYSTEM,DET_P5.format(note=row['note'][:18000],question=row['question'],answer=row['answer'][:2000]),port,900,temp)
    parsed=parse_detection(raw); parsed['valid']=valid_detection(parsed)
    return {'raw':raw,'parsed':parsed,'prompt':'p5_retrieval_payload','temperature':temp}

def run_correction(row,det,spans,example,port,temp)->dict[str,Any]:
    user=COR_USER.format(note=row['note'][:18000],question=row['question'],answer=row['answer'][:1800],error_type=det.get('error_type',''),question_focus=det.get('question_focus',''),wrong_claim=det.get('wrong_claim',''),correct_or_missing_info=det.get('correct_or_missing_info',''),correction_hint=det.get('correction_hint',''),spans_block=render_spans(spans),example_block=render_example(example))
    ans=vllm_chat(COR_SYSTEM,user,port,600,temp)
    return {'answer':ans,'temperature':temp,'raicl_example':example,'spans':spans}

def run_verdict(row,corr_answer,port,k,temp)->dict[str,Any]:
    rng=random.Random(42+(row['fold']<<16)+row['idx']); orig_a=rng.random()>0.5
    ans_a=row['answer'] if orig_a else corr_answer; ans_b=corr_answer if orig_a else row['answer']
    samples=[]
    for _ in range(k):
        raw=vllm_chat(VERDICT_SYSTEM,VERDICT_USER.format(note=row['note'][:18000],question=row['question'],answer_a=ans_a[:1500],answer_b=ans_b[:1500]),port,220,temp)
        parsed=parse_verdict(raw); samples.append({'raw':raw,**parsed})
    counts=Counter(s['pick'] for s in samples); corrected_slot='B' if orig_a else 'A'; corrected_votes=counts.get(corrected_slot,0)
    a,b=counts.get('A',0),counts.get('B',0)
    majority='TIE' if a==b else ('A' if a>b else 'B')
    accept=(majority==corrected_slot and corrected_votes>k/2)
    return {'variant':'pairwise_gate','k':k,'temperature':temp,'orig_in_slot_A':orig_a,'corrected_slot':corrected_slot,'votes':dict(counts),'majority_pick':majority,'accept_correction':accept,'samples':samples}

def process_one(row,port,args)->dict[str,Any]:
    out={k:row[k] for k in ['fold','idx','patient_id','question','ground_truth','answer','orig_label']}
    try:
        det=run_detect(row,port,args.det_temperature); out['detection']=det
        p=det['parsed']
        if p.get('verdict')!='INCORRECT' or not p.get('valid'):
            out['action']='kept_original_no_detection'; out['final_answer']=row['answer']; return out
        spans=retrieve_spans(row,p,args.k_spans); ex=retrieve_example(row,p)
        corr=run_correction(row,p,spans,ex,port,args.correction_temperature); out['correction']=corr
        verdict=run_verdict(row,corr['answer'],port,args.verdict_k,args.verdict_temperature); out['verdict']=verdict
        if verdict['accept_correction']:
            out['action']='accepted_correction'; out['final_answer']=corr['answer']
        else:
            out['action']='rejected_by_verdict'; out['final_answer']=row['answer']
        return out
    except Exception as e:
        out['error']=str(e); out['action']='error_keep_original'; out['final_answer']=row['answer']; return out


def summarize(rows:list[dict[str,Any]])->dict[str,Any]:
    judged=[r for r in rows if (r.get('judge_final') or {}).get('label') is not None]
    fixes=sum(1 for r in judged if r['orig_label']==0 and r['judge_final']['label']==1)
    breaks=sum(1 for r in judged if r['orig_label']==1 and r['judge_final']['label']==0)
    return {'n':len(rows),'n_judged':len(judged),'actions':dict(Counter(r.get('action') for r in rows)),'detected':sum(1 for r in rows if ((r.get('detection') or {}).get('parsed') or {}).get('verdict')=='INCORRECT'),'accepted':sum(1 for r in rows if r.get('action')=='accepted_correction'),'fixes':fixes,'breaks':breaks,'net':fixes-breaks,'parse_paths':dict(Counter(((r.get('detection') or {}).get('parsed') or {}).get('parse_path','none') for r in rows)),'errors':sum(1 for r in rows if r.get('error'))}

def write_jsonl(path:Path,rows:list[dict[str,Any]]):
    with path.open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--port',type=int,default=8003); ap.add_argument('--concurrency',type=int,default=8)
    ap.add_argument('--n-wrong',type=int,default=2); ap.add_argument('--n-correct',type=int,default=2); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--det-temperature',type=float,default=0.7); ap.add_argument('--correction-temperature',type=float,default=0.0); ap.add_argument('--verdict-temperature',type=float,default=0.7)
    ap.add_argument('--verdict-k',type=int,default=3); ap.add_argument('--k-spans',type=int,default=5); ap.add_argument('--judge',action='store_true')
    args=ap.parse_args(); served=served_model_id(args.port)
    if 'qwen2.5' not in served.lower() and 'qwen2' not in served.lower(): raise RuntimeError(f'Expected Qwen2.5, found {served}')
    sample=load_rows(args.n_wrong,args.n_correct,args.seed)
    run_id=f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}_detp5_raicl_vgate"
    out_dir=OUT_ROOT/run_id; out_dir.mkdir(parents=True,exist_ok=True)
    print(f'served_model={served} sample={len(sample)} c={args.concurrency}',flush=True)
    # Warm GTR on first item if needed later.
    if sample: topk_spans(sample[0]['note'],[sample[0]['question']],k=1,scoring='agreement')
    rows=[]
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs=[ex.submit(process_one,row,args.port,args) for row in sample]
        for i,fut in enumerate(as_completed(futs),1):
            rows.append(fut.result()); print(f'pipeline {i}/{len(futs)}',flush=True)
    write_jsonl(out_dir/'pipeline_outputs.jsonl',rows)
    if args.judge:
        note_by_key={(r['fold'],r['idx']):r['note'] for r in sample}
        for i,r in enumerate(rows,1):
            note=note_by_key[(r['fold'],r['idx'])]
            r['judge_final']=gpt_judge(note,r['question'],r['ground_truth'],r['final_answer'])
            if i%10==0 or i==len(rows): print(f'judged {i}/{len(rows)}',flush=True)
        write_jsonl(out_dir/'judged_outputs.jsonl',rows)
    summary={'task':'self_detection_raicl_correction_verdict','served_model':served,'settings':vars(args),'summary':summarize(rows),'outputs':{'pipeline':str(out_dir/'pipeline_outputs.jsonl'),'judged':str(out_dir/'judged_outputs.jsonl') if args.judge else None}}
    (out_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())

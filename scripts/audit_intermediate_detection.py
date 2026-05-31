#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path(os.environ.get('MED_HEAL_SOURCE_REPO', PROJECT_ROOT.parent / 'llm-ehr-hallucination'))

SYSTEM = "You are a strict clinical QA auditor. You audit whether an intermediate detection/correction diagnosis is valid."


def load_api_key() -> str:
    for env in (PROJECT_ROOT/'.env', SOURCE_REPO/'.env'):
        if env.exists():
            for line in env.read_text().splitlines():
                line=line.strip()
                if line.startswith('OPENAI_API_KEY=') and not line.startswith('#'):
                    return line.split('=',1)[1].strip()
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    raise RuntimeError('OPENAI_API_KEY not found')

_client: OpenAI|None = None

def client() -> OpenAI:
    global _client
    if _client is None:
        _client=OpenAI(api_key=load_api_key())
    return _client


def audit_user(row:dict[str,Any], note:str)->str:
    det=row.get('detection') or {}
    parsed=det.get('parsed') or {}
    corr=(row.get('correction') or {}).get('answer') or ''
    verdict=row.get('verdict') or {}
    return f"""DISCHARGE SUMMARY:
{note}

QUESTION:
{row.get('question','')}

GROUND TRUTH ANSWER:
{row.get('ground_truth','')}

ZERO-SHOT ANSWER:
{row.get('answer','')}

RAW DETECTION / AUDIT MEMO:
{det.get('raw','')}

PARSED DETECTION FIELDS:
{json.dumps(parsed, ensure_ascii=False, indent=2)}

CORRECTION CANDIDATE, IF PRODUCED:
{corr or '(none)'}

VERDICT/GATE OUTPUT, IF PRODUCED:
{json.dumps(verdict, ensure_ascii=False)[:4000] if verdict else '(none)'}

Task: audit the intermediate diagnosis, not just the final answer. Decide whether the detection/correction stage correctly identified a real correction-worthy issue and whether the proposed correction follows from it.

Return JSON only with this schema:
{{
  "detection_valid": "YES|NO|UNCLEAR",
  "error_type_valid": "YES|NO|UNCLEAR",
  "wrong_claim_valid": "YES|NO|UNCLEAR",
  "correction_target_supported": "YES|NO|UNCLEAR",
  "correction_operation_safe": "YES|NO|UNCLEAR",
  "evidence_sufficient": "YES|NO|UNCLEAR",
  "correction_candidate_matches_detection": "YES|NO|UNCLEAR|NO_CANDIDATE",
  "stage_failure": "NONE|DETECTION|PARSING|CORRECTION|VERDICT_GATE|UNCLEAR",
  "reason": "one concise explanation"
}}
"""


def gpt4o_audit(row:dict[str,Any], note:str)->dict[str,Any]:
    for attempt in range(5):
        try:
            r=client().chat.completions.create(
                model='gpt-4o',
                messages=[{'role':'system','content':SYSTEM},{'role':'user','content':audit_user(row,note)}],
                temperature=0.0,
                max_tokens=450,
                response_format={'type':'json_object'},
            )
            raw=(r.choices[0].message.content or '{}').strip()
            out=json.loads(raw)
            out['raw']=raw
            out['model']='gpt-4o'
            return out
        except Exception as e:
            if attempt==4:
                return {'error':str(e),'model':'gpt-4o'}
            time.sleep(2*(attempt+1))
    return {'error':'unknown'}


def load_jsonl(path:Path)->list[dict[str,Any]]:
    rows=[]
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path:Path, rows:list[dict[str,Any]])->None:
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')


def summarize(rows:list[dict[str,Any]])->dict[str,Any]:
    audits=[r.get('intermediate_audit') or {} for r in rows]
    return {
        'n':len(rows),
        'audited':sum(1 for a in audits if a and not a.get('error')),
        'detection_valid':dict(Counter(a.get('detection_valid','MISSING') for a in audits)),
        'error_type_valid':dict(Counter(a.get('error_type_valid','MISSING') for a in audits)),
        'wrong_claim_valid':dict(Counter(a.get('wrong_claim_valid','MISSING') for a in audits)),
        'correction_target_supported':dict(Counter(a.get('correction_target_supported','MISSING') for a in audits)),
        'correction_operation_safe':dict(Counter(a.get('correction_operation_safe','MISSING') for a in audits)),
        'evidence_sufficient':dict(Counter(a.get('evidence_sufficient','MISSING') for a in audits)),
        'correction_candidate_matches_detection':dict(Counter(a.get('correction_candidate_matches_detection','MISSING') for a in audits)),
        'stage_failure':dict(Counter(a.get('stage_failure','MISSING') for a in audits)),
        'pipeline_actions':dict(Counter(r.get('action') for r in rows)),
        'detected_by_pipeline':sum(1 for r in rows if ((r.get('detection') or {}).get('parsed') or {}).get('verdict')=='INCORRECT'),
        'accepted_by_pipeline':sum(1 for r in rows if r.get('action')=='accepted_correction'),
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--only-detected', action='store_true')
    args=ap.parse_args()
    in_path=args.run_dir/'judged_outputs.jsonl'
    if not in_path.exists():
        in_path=args.run_dir/'pipeline_outputs.jsonl'
    rows=load_jsonl(in_path)
    for i,r in enumerate(rows,1):
        if args.only_detected and ((r.get('detection') or {}).get('parsed') or {}).get('verdict')!='INCORRECT':
            r['intermediate_audit']={'skipped':'not_detected'}
        else:
            r['intermediate_audit']=gpt4o_audit(r, r.get('note',''))
        if i%10==0 or i==len(rows):
            print(f'audited {i}/{len(rows)}', flush=True)
    write_jsonl(args.run_dir/'intermediate_audit.jsonl', rows)
    summary=summarize(rows)
    (args.run_dir/'intermediate_audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

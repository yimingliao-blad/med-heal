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

PROJECT_ROOT=Path(__file__).resolve().parents[1]
SOURCE_REPO=Path(os.environ.get('MED_HEAL_SOURCE_REPO', PROJECT_ROOT.parent / 'llm-ehr-hallucination'))

SYSTEM="""You are a faithful parser between pipeline stages. You do not make an independent medical judgment. You read a natural-language audit memo from the tested model and convert it into the exact structured payload needed by the next correction stage."""

SCHEMA="""Return JSON only:
{
  "verdict":"CORRECT|INCORRECT|UNCLEAR",
  "error_type":"CONTRADICTION|OMISSION|QUESTION_MISALIGNMENT|NONE|UNCLEAR",
  "question_type":"string",
  "required_answer_format":"string",
  "question_focus":"string",
  "answer_focus":"string",
  "slot_check":"supported|full-contradiction|partial-conflict|missing-central-slot|wrong-focus|sufficient|string",
  "key_evidence_reason":"string",
  "full_contradiction":"YES|NO|UNCLEAR",
  "correction_operation":"REPLACE_VALUE|ADD_MISSING_SLOT|REMOVE_UNSUPPORTED_CLAIM|REFOCUS_TIME_OR_VISIT|KEEP_ORIGINAL",
  "evidence_sufficient_for_correction":"YES|NO|UNCLEAR",
  "decisive_evidence":"string or NONE",
  "do_not_change":"string or NONE",
  "wrong_claim":"string or NONE",
  "correct_or_missing_info":"string or NONE",
  "evidence_needed":"string or NONE",
  "retrieval_queries":["string"],
  "correction_hint":"string or NONE",
  "why":"string"
}
"""


def load_api_key()->str:
    for env in (PROJECT_ROOT/'.env', SOURCE_REPO/'.env'):
        if env.exists():
            for line in env.read_text().splitlines():
                line=line.strip()
                if line.startswith('OPENAI_API_KEY=') and not line.startswith('#'):
                    return line.split('=',1)[1].strip()
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    raise RuntimeError('OPENAI_API_KEY not found')

_client:OpenAI|None=None

def client()->OpenAI:
    global _client
    if _client is None:
        _client=OpenAI(api_key=load_api_key())
    return _client


def user_prompt(row:dict[str,Any])->str:
    det=row.get('detection') or {}
    plan=det.get('plan_raw') or ''
    raw=det.get('raw') or ''
    return f"""Question:
{row.get('question','')}

Zero-shot answer:
{row.get('answer','')}

Natural audit plan:
{plan}

Natural audit memo to parse:
{raw}

Parsing rules:
- Do not re-audit from the note or ground truth. Interpret the memo faithfully.
- If the memo says the answer is fully supported and no change is needed, emit verdict CORRECT, error_type NONE, correction_operation KEEP_ORIGINAL.
- If the memo says any required answer component is missing, should be added, should be included for completeness, or needs even a small correction to fully answer the question, emit verdict INCORRECT with error_type OMISSION unless it is clearly contradiction or wrong-focus.
- If the memo identifies a wrong value/date/medication/fact, emit verdict INCORRECT with error_type CONTRADICTION and operation REPLACE_VALUE.
- If the memo says the answer addresses the wrong visit, date, time window, or aspect, emit verdict INCORRECT with error_type QUESTION_MISALIGNMENT and operation REFOCUS_TIME_OR_VISIT.
- Use exactly one correction_operation enum. Use ADD_MISSING_SLOT for missing required information. Use KEEP_ORIGINAL only when no correction is recommended.
- evidence_sufficient_for_correction should be YES only when the memo names note evidence or enough source support for the correction.
- Preserve supported original content in do_not_change.
- The downstream correction stage relies on wrong_claim, correct_or_missing_info, evidence_needed, decisive_evidence, and correction_hint. Fill them concretely when verdict is INCORRECT.

{SCHEMA}"""


def parse_row(row:dict[str,Any])->dict[str,Any]:
    for attempt in range(5):
        try:
            r=client().chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role':'system','content':SYSTEM},{'role':'user','content':user_prompt(row)}],
                temperature=0.0,
                max_tokens=650,
                response_format={'type':'json_object'},
            )
            raw=(r.choices[0].message.content or '{}').strip()
            obj=json.loads(raw)
            obj['raw_parser_output']=raw
            obj['parse_path']='gpt4o-mini-helper-v2'
            return obj
        except Exception as e:
            if attempt==4:
                return {'error':str(e),'parse_path':'gpt4o-mini-helper-v2'}
            time.sleep(2*(attempt+1))
    return {'error':'unknown','parse_path':'gpt4o-mini-helper-v2'}


def norm(s:Any)->str:
    return str(s or '').strip().upper()


def valid_for_correction(p:dict[str,Any])->bool:
    if norm(p.get('verdict')) == 'CORRECT':
        return False
    if norm(p.get('verdict')) != 'INCORRECT':
        return False
    if norm(p.get('error_type')) not in {'CONTRADICTION','OMISSION','QUESTION_MISALIGNMENT'}:
        return False
    if norm(p.get('correction_operation')) not in {'REPLACE_VALUE','ADD_MISSING_SLOT','REMOVE_UNSUPPORTED_CLAIM','REFOCUS_TIME_OR_VISIT'}:
        return False
    if not (str(p.get('wrong_claim') or '').strip() or str(p.get('correct_or_missing_info') or '').strip()):
        return False
    if not (str(p.get('evidence_needed') or '').strip() or str(p.get('decisive_evidence') or '').strip() or p.get('retrieval_queries')):
        return False
    return True


def load_jsonl(path:Path)->list[dict[str,Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path:Path, rows:list[dict[str,Any]])->None:
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    args=ap.parse_args()
    in_path=args.run_dir/'judged_outputs.jsonl'
    if not in_path.exists():
        in_path=args.run_dir/'pipeline_outputs.jsonl'
    rows=load_jsonl(in_path)
    out=[]
    for i,r in enumerate(rows,1):
        parsed=parse_row(r)
        parsed['valid_for_correction']=valid_for_correction(parsed)
        r2=dict(r)
        r2['reparsed_detection']=parsed
        out.append(r2)
        if i%10==0 or i==len(rows):
            print(f'reparsed {i}/{len(rows)}', flush=True)
    write_jsonl(args.run_dir/'reparsed_mini_helper_v2.jsonl', out)
    parsed=[r['reparsed_detection'] for r in out]
    summary={
        'n':len(out),
        'verdicts':dict(Counter(norm(p.get('verdict')) for p in parsed)),
        'error_types':dict(Counter(norm(p.get('error_type')) for p in parsed)),
        'operations':dict(Counter(norm(p.get('correction_operation')) for p in parsed)),
        'valid_for_correction':sum(1 for p in parsed if p.get('valid_for_correction')),
        'parse_path':dict(Counter(p.get('parse_path') for p in parsed)),
        'errors':sum(1 for p in parsed if p.get('error')),
    }
    (args.run_dir/'reparsed_mini_helper_v2_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

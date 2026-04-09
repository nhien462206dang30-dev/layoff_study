"""
Improved AI-mention labeling using Claude API
==============================================
The original regex-based approach only processed 89 articles (most URLs
paywalled or blocked) and found 18 AI-motivated events.

This script improves coverage by asking Claude to assess each event based
on its knowledge of publicly reported tech layoff announcements.

Strategy:
  - Focus on post-GenAI events (>=2023) where AI motivation is plausible
  - Batch 30 events per API call to keep costs low
  - Preserve existing STRONG evidence labels; re-evaluate WEAK and unlabeled
  - Use 3-tier confidence: high / medium / low
  - Only mark ai_mentioned=1 if confidence is high or medium

Output: updates master_events_final.csv in place + saves audit log
"""

import os
import re
import json
import time
import subprocess
import pandas as pd
from pathlib import Path

BASE          = Path('/Users/irmina/Documents/Claude/layoff_study')
EVENTS_PATH   = BASE / 'data/processed/master_events_final.csv'
AUDIT_PATH    = BASE / 'data/processed/ai_label_audit.csv'

BATCH = 15


def call_claude_cli(prompt: str) -> str:
    """Call Claude via the claude CLI (uses existing Claude Code auth)."""
    result = subprocess.run(
        ['claude', '-p', prompt],
        capture_output=True, text=True, timeout=180
    )
    if result.returncode != 0:
        raise RuntimeError(f'claude CLI error: {result.stderr[:200]}')
    return result.stdout.strip()


def build_prompt(batch_info: list[dict]) -> str:
    return f"""You are a financial research assistant with knowledge of tech industry news.

For each tech company layoff event below, determine whether the company's layoff announcement
publicly cited or was widely reported as motivated by:
  - AI / generative AI adoption replacing workers
  - Automation or machine learning reducing headcount
  - Restructuring to invest in AI (pivoting budget toward AI teams)
  - Efficiency gains from AI tools reducing need for certain roles

Use your knowledge of publicly reported news and company statements up to your training cutoff.

Events:
{json.dumps(batch_info, indent=2)}

For each event, return a JSON object with this exact structure:
{{
  "<idx>": {{
    "ai_mentioned": 0 or 1,
    "confidence": "high" | "medium" | "low",
    "reasoning": "one sentence explaining your assessment"
  }},
  ...
}}

Rules:
- ai_mentioned=1 only if the layoff was publicly and clearly linked to AI/automation investment or replacement
- "high" = company explicitly stated AI as reason in press release or CEO quote
- "medium" = multiple credible news sources linked the layoff to AI strategy
- "low" = speculative or only indirect connection
- For pre-2022 events, be very conservative — AI motivation was rare before ChatGPT
- Return ONLY valid JSON, no other text.
"""


def call_claude(batch_info):
    prompt = build_prompt(batch_info)
    try:
        text = call_claude_cli(prompt)
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f'  Claude CLI error: {e}')
    return {}


def main():
    print('=' * 60)
    print('IMPROVED AI-MENTION LABELING VIA CLAUDE API')
    print('=' * 60)

    df = pd.read_csv(EVENTS_PATH)
    df['announcement_date'] = pd.to_datetime(df['announcement_date'])
    print(f'Loaded {len(df)} events')
    print(f'  Current ai_mentioned=1: {df["ai_mentioned"].sum()}')

    # ── Select events to (re)label ─────────────────────────────────────────
    # Re-label all events EXCEPT those with STRONG evidence (keep those as-is)
    has_strong = df['ai_evidence'].fillna('').str.contains(r'\[STRONG\]', regex=True)
    to_label   = df[~has_strong].copy()

    # Prioritize post-GenAI events but include all
    post = to_label[to_label['announcement_date'].dt.year >= 2023]
    pre  = to_label[to_label['announcement_date'].dt.year <  2023]
    to_label = pd.concat([post, pre]).reset_index(drop=True)

    print(f'\nEvents to re-label: {len(to_label)} '
          f'({len(post)} post-GenAI, {len(pre)} pre-GenAI)')
    print(f'Keeping {has_strong.sum()} events with existing STRONG evidence unchanged\n')

    # ── Batch labeling ──────────────────────────────────────────────────────
    all_labels = {}   # original df index -> {ai_mentioned, confidence, reasoning}

    for batch_start in range(0, len(to_label), BATCH):
        chunk = to_label.iloc[batch_start: batch_start + BATCH]

        batch_info = []
        for _, row in chunk.iterrows():
            batch_info.append({
                'idx':      str(row.name),
                'company':  row['company_fyi'],
                'date':     str(row['announcement_date'])[:10],
                'industry': str(row.get('industry', '')),
            })

        results = call_claude(batch_info)
        all_labels.update(results)

        done  = min(batch_start + BATCH, len(to_label))
        pos   = sum(1 for v in results.values() if v.get('ai_mentioned') == 1)
        himed = sum(1 for v in results.values()
                    if v.get('ai_mentioned') == 1
                    and v.get('confidence') in ('high', 'medium'))
        print(f'  Batch {batch_start//BATCH + 1}: '
              f'{done}/{len(to_label)} done | '
              f'{pos} AI-positive ({himed} high/medium confidence)')

        time.sleep(0.5)

    # ── Apply labels back ──────────────────────────────────────────────────
    # Only set ai_mentioned=1 for high/medium confidence
    audit_rows = []
    changed = 0

    for idx_str, label in all_labels.items():
        try:
            orig_idx = int(idx_str)
        except ValueError:
            continue
        if orig_idx not in df.index:
            continue

        new_val    = label.get('ai_mentioned', 0)
        conf       = label.get('confidence', 'low')
        reasoning  = label.get('reasoning', '')
        old_val    = int(df.at[orig_idx, 'ai_mentioned'])

        # Apply only high/medium confidence labels
        if conf in ('high', 'medium'):
            apply_val = new_val
        else:
            apply_val = 0   # treat low-confidence AI=1 as 0

        audit_rows.append({
            'orig_idx':    orig_idx,
            'company':     df.at[orig_idx, 'company_fyi'],
            'date':        df.at[orig_idx, 'announcement_date'],
            'old_label':   old_val,
            'new_label':   apply_val,
            'confidence':  conf,
            'reasoning':   reasoning,
            'changed':     old_val != apply_val,
        })

        if old_val != apply_val:
            df.at[orig_idx, 'ai_mentioned']  = apply_val
            df.at[orig_idx, 'ai_evidence']   = f'[CLAUDE-{conf.upper()}] {reasoning}'
            changed += 1

    # ── Save ───────────────────────────────────────────────────────────────
    df.to_csv(EVENTS_PATH, index=False)
    print(f'\nSaved updated events → {EVENTS_PATH}')

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(AUDIT_PATH, index=False)
    print(f'Audit log saved → {AUDIT_PATH}')

    # ── Summary ────────────────────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('LABELING SUMMARY')
    print(f'{"=" * 60}')
    print(f'  Total events processed : {len(all_labels)}')
    print(f'  Labels changed         : {changed}')
    print(f'  Final ai_mentioned=1   : {df["ai_mentioned"].sum()}  '
          f'(was {df["ai_mentioned"].sum() - changed + (audit_df["old_label"] == 1).sum() - has_strong.sum()})')
    print(f'  Post-GenAI ai=1        : '
          f'{((df["ai_mentioned"]==1) & (df["announcement_date"].dt.year >= 2023)).sum()}')
    print(f'  Pre-GenAI  ai=1        : '
          f'{((df["ai_mentioned"]==1) & (df["announcement_date"].dt.year < 2023)).sum()}')

    if len(audit_df) > 0:
        print(f'\nTop newly labeled AI events (confidence=high/medium, new=1):')
        new_pos = audit_df[(audit_df['new_label'] == 1) &
                           (audit_df['old_label'] == 0) &
                           (audit_df['confidence'].isin(['high','medium']))]
        new_pos = new_pos.sort_values('confidence').head(20)
        for _, r in new_pos.iterrows():
            print(f'  {r["company"]:<25} {str(r["date"])[:10]}  [{r["confidence"]}] {r["reasoning"][:80]}')


if __name__ == '__main__':
    main()

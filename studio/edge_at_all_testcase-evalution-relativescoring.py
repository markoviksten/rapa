#!/usr/bin/env python3
"""
Edge et al. (2024) -menetelmä – LLM-as-Judge parivertailu
Korvaa RAGAS-evaluoinnin tilanteissa, joissa ground truthia ei ole.

Arviointidimensiot:
  - Kattavuus     (Comprehensiveness)
  - Monipuolisuus (Diversity)
  - Oivalluttavuus (Empowerment)

Pisteytys: suhteellinen jako – score_a + score_b = 100.0 %
  50/50 = tasaväkinen, >55 = selkeä etu, voittaja jos oma osuus > 55 %

Testikysymykset luetaan generate_testcases_graphrag.py:n tuottamasta JSON:sta
(ei ground_truth-kenttää, vain question + metadata).

Käyttö:
  python run_edge_eval.py
  python run_edge_eval.py --testfile studio/testcases/tc_20260212.json
  python run_edge_eval.py --mode-a mix --mode-b hybrid
"""

import os
import sys
import json
import random
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================================
# KONFIGURAATIO
# ============================================================================

OPENAI_API_KEY     = os.getenv("LLM_BINDING_API_KEY")
LIGHTRAG_ENDPOINT  = "http://localhost:9623"
LIGHTRAG_JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6NDkyMzgxMzIzMCwicm9sZSI6InVzZXIiLCJtZXRhZGF0YSI6eyJhdXRoX21vZGUiOiJlbmFibGVkIn19.0BTuVkGRO2wljpwiZoKmob22_Z7FiIj0VdqgPsKMHh4"
LIGHTRAG_API_KEY   = "your-secure-api-key-here-marko"

# Vertailtavat modet: A vs B
MODE_A = "naive"
MODE_B = "mix"

# Testikysymystiedosto
TEST_FILE = "studio/testcases/tc_20260227_080328.json"

# Arvioija-LLM
JUDGE_MODEL = "gpt-4o"   # Suositellaan isompaa mallia arviointiin

# RAG-hakuasetukset
TOP_K      = 10   # naive-moden top_k
USER_PROMPT = ""

# Parametrit graph-pohjaisia modeja varten (kaikki muut kuin naive)
GRAPH_MODE_PARAMS = {
    "top_k":               20,
    "chunk_top_k":         40,
    "max_entity_tokens":   6000,
    "max_relation_tokens": 8000,
    "max_total_tokens":    30000,
    "enable_rerank":       True,
}

# Satunnaista A/B-järjestystä position biasin vähentämiseksi
RANDOMIZE_AB_ORDER = True

# Token-hinnat
TOKEN_PRICES = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
}

# ============================================================================
# Älä muokkaa tästä alaspäin
# ============================================================================

if not OPENAI_API_KEY:
    raise ValueError("LLM_BINDING_API_KEY ei löydy .env-tiedostosta!")

client = OpenAI(api_key=OPENAI_API_KEY)

# Hakemistot
script_dir   = Path(__file__).parent.absolute()
project_root = script_dir
while project_root.name and not ((project_root / "data").exists() or (project_root / "studio").exists()):
    if project_root.parent == project_root:
        break
    project_root = project_root.parent
if not ((project_root / "data").exists() or (project_root / "studio").exists()):
    project_root = script_dir

output_dir = project_root / "studio" / "testresults"
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# APUFUNKTIOT – LIGHTRAG-HAKU
# ============================================================================

def query_lightrag(question: str, mode: str) -> Tuple[str, List[str]]:
    """Hae vastaus LightRAG:lta. Palauttaa (vastaus, kontekstit)."""

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LIGHTRAG_JWT_TOKEN}",
        "X-API-Key": LIGHTRAG_API_KEY,
    }

    is_naive = mode == "naive"
    payload = {
        "query": question,
        "mode": mode,
        "top_k": TOP_K if is_naive else GRAPH_MODE_PARAMS["top_k"],
        "only_need_context": False,
        "include_references": True,
        "include_chunk_content": True,
        "stream": False,
    }
    if not is_naive:
        payload.update({
            "chunk_top_k":         GRAPH_MODE_PARAMS["chunk_top_k"],
            "max_entity_tokens":   GRAPH_MODE_PARAMS["max_entity_tokens"],
            "max_relation_tokens": GRAPH_MODE_PARAMS["max_relation_tokens"],
            "max_total_tokens":    GRAPH_MODE_PARAMS["max_total_tokens"],
            "enable_rerank":       GRAPH_MODE_PARAMS["enable_rerank"],
        })
    if USER_PROMPT:
        payload["user_prompt"] = USER_PROMPT

    try:
        r = requests.post(
            f"{LIGHTRAG_ENDPOINT}/query",
            json=payload,
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()

        answer = data.get("response", "")

        # Kontekstit
        refs = data.get("references", [])
        contexts = []
        for ref in refs:
            if isinstance(ref, str):
                contexts.append(ref.strip())
            elif isinstance(ref, dict):
                text = (ref.get("chunk_content") or
                        ref.get("content") or
                        ref.get("text") or "")
                if isinstance(text, list):
                    text = "\n".join(str(t) for t in text)
                if text and text.strip():
                    contexts.append(text.strip())

        return answer, contexts

    except Exception as e:
        print(f"    ⚠️  LightRAG-virhe (mode={mode}): {e}")
        return "", []


# ============================================================================
# LLM-AS-JUDGE – PARIVERTAILU
# ============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of AI-generated answers.
You will compare two answers (Answer A and Answer B) to the same question.
Be objective and base your judgment solely on the content of the answers.
Always respond with valid JSON only."""


def build_judge_prompt(question: str, answer_a: str, answer_b: str, dimension: str, dimension_fi: str, dimension_desc: str) -> str:
    return f"""Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Evaluate both answers based on **{dimension} ({dimension_fi})**: {dimension_desc}

Step 1 – Distribute 100 points between the two answers to reflect their RELATIVE quality.
  The two scores must sum to exactly 100.00.
  Reference scale (score_a / score_b):
  - 50.00 / 50.00  → equally good
  - 55.00 / 45.00  → Answer A is noticeably better
  - 60.00 / 40.00  → Answer A is clearly better
  - 70.00 / 30.00  → Answer A is substantially better
  - (mirror values apply when Answer B is better)

Step 2 – Decide the winner:
  - "A" if score_a > 55.00  (Answer A has a meaningful edge)
  - "B" if score_b > 55.00  (Answer B has a meaningful edge)
  - "tie" if both scores are within the range 45.00–55.00

Respond ONLY with this JSON (score_a + score_b MUST equal 100.00):
{{
  "score_a": <float, e.g. 62.50>,
  "score_b": <float, e.g. 37.50>,
  "winner": "A" | "B" | "tie",
  "reasoning": "Brief explanation in Finnish (1-2 sentences)"
}}"""


DIMENSIONS = [
    {
        "key":   "kattavuus",
        "en":    "Comprehensiveness",
        "fi":    "Kattavuus",
        "desc":  "How thoroughly does the answer address all aspects of the question?",
    },
    {
        "key":   "monipuolisuus",
        "en":    "Diversity",
        "fi":    "Monipuolisuus",
        "desc":  "How diverse and varied are the perspectives and information presented?",
    },
    {
        "key":   "oivalluttavuus",
        "en":    "Empowerment",
        "fi":    "Oivalluttavuus",
        "desc":  "How well does the answer help the reader understand the topic and form their own conclusions?",
    },
]


def _normalize_to_100(raw_a: float, raw_b: float) -> Tuple[float, float]:
    """
    Varmistaa että score_a + score_b = 100.0.
    Jos LLM on palauttanut arvot jotka eivät summaudu 100:aan, normalisoidaan.
    """
    total = raw_a + raw_b
    if total <= 0:
        return 50.0, 50.0
    factor = 100.0 / total
    a = round(raw_a * factor, 2)
    b = round(100.0 - a, 2)   # varmistaa täsmälleen 100.0
    return a, b


def _winner_from_scores(score_a: float, score_b: float) -> str:
    """Päättele voittaja suhteellisista pisteistä (yhteensä 100)."""
    if score_a > 55.0:
        return "A"
    elif score_b > 55.0:
        return "B"
    else:
        return "tie"


def judge_pair(question: str, answer_a: str, answer_b: str, swapped: bool) -> Dict[str, Any]:
    """
    Arvioi vastauspari kolmella dimensiolla.
    Palauttaa suhteelliset pisteet joiden summa on 100.0 per dimensio.
    Jos swapped=True, A ja B ovat vaihtaneet paikkaa – korjataan tulos.
    """
    judgments = {}
    total_tokens = {"input": 0, "output": 0}

    for dim in DIMENSIONS:
        prompt = build_judge_prompt(
            question, answer_a, answer_b,
            dim["en"], dim["fi"], dim["desc"]
        )

        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=250,
                temperature=0,
                response_format={"type": "json_object"},
            )

            total_tokens["input"]  += resp.usage.prompt_tokens
            total_tokens["output"] += resp.usage.completion_tokens

            j = json.loads(resp.choices[0].message.content)

            raw_a = float(j.get("score_a", 50.0))
            raw_b = float(j.get("score_b", 50.0))

            # Normalisoi summaksi 100.0
            norm_a, norm_b = _normalize_to_100(raw_a, raw_b)

            reasoning = j.get("reasoning", "")

            if swapped:
                # Promptissa A = mode_b, B = mode_a → vaihdetaan pisteet takaisin
                score_a = norm_b   # mode_a:n osuus
                score_b = norm_a   # mode_b:n osuus
            else:
                score_a = norm_a
                score_b = norm_b

            # Johdetaan voittaja normalisoiduista pisteistä (ei LLM:n ehdotuksesta),
            # jotta winner on aina johdonmukainen pistemäärien kanssa.
            winner = _winner_from_scores(score_a, score_b)

            judgments[dim["key"]] = {
                "winner":    winner,
                "score_a":   score_a,
                "score_b":   score_b,
                "reasoning": reasoning,
            }

        except Exception as e:
            print(f"      ⚠️  Arviointivirhe ({dim['fi']}): {e}")
            judgments[dim["key"]] = {
                "winner":    "tie",
                "score_a":   50.0,
                "score_b":   50.0,
                "reasoning": f"Virhe: {e}",
            }

        time.sleep(0.3)

    return judgments, total_tokens


# ============================================================================
# PÄÄOHJELMA
# ============================================================================

def main(test_file: str, mode_a: str, mode_b: str):

    print("=" * 80)
    print("EDGE ET AL. (2024) – LLM-AS-JUDGE PARIVERTAILU")
    print("=" * 80)
    print(f"Mode A:      {mode_a}")
    print(f"Mode B:      {mode_b}")
    print(f"Arvioija:    {JUDGE_MODEL}")
    print(f"Testit:      {test_file}")
    print(f"Output:      {output_dir}")
    print(f"Pisteytys:   Suhteellinen – A-osuus + B-osuus = 100 %")
    print("=" * 80 + "\n")

    # ── Lataa testikysymykset ──────────────────────────────────────────────
    file_path = project_root / test_file
    if not file_path.exists():
        file_path = Path(test_file)
    if not file_path.exists():
        print(f"❌ Tiedostoa ei löydy: {test_file}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    if not test_cases:
        print("❌ Ei testikysymyksiä löydetty tiedostosta.")
        sys.exit(1)

    print(f"✓ Ladattu {len(test_cases)} kysymystä\n")

    # ── Ajosilmukka ───────────────────────────────────────────────────────
    results      = []
    total_tokens = {"input": 0, "output": 0}
    start_time   = time.time()

    # Voittolaskurit ja pistekertymät dimensioittain
    wins   = {d["key"]: {"A": 0, "B": 0, "tie": 0} for d in DIMENSIONS}
    scores = {d["key"]: {"A": [], "B": []} for d in DIMENSIONS}

    for idx, tc in enumerate(test_cases, 1):
        question = tc.get("question", "")
        meta     = tc.get("metadata", {})

        print(f"[{idx}/{len(test_cases)}] {question[:80]}...")

        # Hae vastaukset molemmilta modeilta
        print(f"  → Haetaan mode={mode_a}...")
        ans_a, ctx_a = query_lightrag(question, mode_a)

        print(f"  → Haetaan mode={mode_b}...")
        ans_b, ctx_b = query_lightrag(question, mode_b)

        # Satunnainen A/B-järjestys position biasin vähentämiseksi
        swapped = RANDOMIZE_AB_ORDER and random.random() < 0.5
        judge_a = ans_b if swapped else ans_a
        judge_b = ans_a if swapped else ans_b

        print(f"  → Arvioidaan ({JUDGE_MODEL}){'  [järjestys vaihdettu]' if swapped else ''}...")
        judgments, q_tokens = judge_pair(question, judge_a, judge_b, swapped)

        total_tokens["input"]  += q_tokens["input"]
        total_tokens["output"] += q_tokens["output"]

        # Kirjaa voitot ja pisteet
        q_score_a_total, q_score_b_total = 0, 0
        for dim_key, j in judgments.items():
            winner = j["winner"]
            wins[dim_key][winner] = wins[dim_key].get(winner, 0) + 1
            scores[dim_key]["A"].append(j["score_a"])
            scores[dim_key]["B"].append(j["score_b"])
            q_score_a_total += j["score_a"]
            q_score_b_total += j["score_b"]

        # Kokonaispisteytys: keskiarvo dimensioiden yli, summa ≈ 100 %
        q_score_a_avg = q_score_a_total / len(DIMENSIONS)
        q_score_b_avg = q_score_b_total / len(DIMENSIONS)

        results.append({
            "question":    question,
            "metadata":    meta,
            "answer_a":    ans_a,
            "answer_b":    ans_b,
            "contexts_a":  ctx_a,
            "contexts_b":  ctx_b,
            "judgments":   judgments,
            "swapped":     swapped,
        })

        # Tulosta dimensiokohtaiset voittajat + pisteet
        for dim in DIMENSIONS:
            j = judgments[dim["key"]]
            w = j["winner"]
            label = f"Mode {w}" if w in ("A", "B") else "Tasapeli"
            print(f"     {dim['fi']:20s}: {label:10s}  A={j['score_a']:5.1f}%  B={j['score_b']:5.1f}%  (yht. {j['score_a']+j['score_b']:.1f}%)")
        print(f"     {'Kokonaispistemäärä':20s}:            A={q_score_a_avg:5.1f}%  B={q_score_b_avg:5.1f}%")
        print()

        time.sleep(0.5)

    duration = time.time() - start_time
    n        = len(test_cases)

    # ── Laske win ratet ja pistekeskiarvot ──────────────────────────────────
    win_rates = {}
    for dim_key, w in wins.items():
        dim_scores_a = scores[dim_key]["A"]
        dim_scores_b = scores[dim_key]["B"]
        win_rates[dim_key] = {
            "A_wins":       w.get("A", 0),
            "B_wins":       w.get("B", 0),
            "ties":         w.get("tie", 0),
            "A_win_rate":   round(w.get("A", 0) / n, 4),
            "B_win_rate":   round(w.get("B", 0) / n, 4),
            "tie_rate":     round(w.get("tie", 0) / n, 4),
            "A_score_avg":  round(sum(dim_scores_a) / len(dim_scores_a), 1) if dim_scores_a else 50.0,
            "B_score_avg":  round(sum(dim_scores_b) / len(dim_scores_b), 1) if dim_scores_b else 50.0,
        }

    # Kokonaispistemäärät: keskiarvo kaikkien dimensioiden yli
    # Summa pysyy ~100 % koska jokainen dimensio on normalisoitu 100:aan
    all_scores_a = [s for d in DIMENSIONS for s in scores[d["key"]]["A"]]
    all_scores_b = [s for d in DIMENSIONS for s in scores[d["key"]]["B"]]
    total_score_a = round(sum(all_scores_a) / len(all_scores_a), 1) if all_scores_a else 50.0
    total_score_b = round(100.0 - total_score_a, 1)   # varmistaa täsmälleen 100.0

    # Kokonaisvoittaja per dimensio
    overall_winner = {}
    for dim_key, wr in win_rates.items():
        if wr["A_wins"] > wr["B_wins"]:
            overall_winner[dim_key] = mode_a
        elif wr["B_wins"] > wr["A_wins"]:
            overall_winner[dim_key] = mode_b
        else:
            overall_winner[dim_key] = "tasapeli"

    # Kokonaisvoittaja pisteiden perusteella
    total_winner = _winner_from_scores(total_score_a, total_score_b)
    total_winner_label = (mode_a if total_winner == "A"
                          else mode_b if total_winner == "B"
                          else "tasapeli")

    # ── Kustannukset ─────────────────────────────────────────────────────
    prices      = TOKEN_PRICES.get(JUDGE_MODEL, {"input": 0, "output": 0})
    input_cost  = (total_tokens["input"]  / 1_000_000) * prices["input"]
    output_cost = (total_tokens["output"] / 1_000_000) * prices["output"]
    total_cost  = input_cost + output_cost

    # ── Tulosta yhteenveto ────────────────────────────────────────────────
    print("=" * 95)
    print("TULOKSET  (suhteellinen pisteytys – A-osuus + B-osuus = 100 %)")
    print("=" * 95)
    print(f"{'Dimensio':<22} {'A win%':>8} {'B win%':>8} {'Tas%':>6}  {'A osuus':>9} {'B osuus':>9}  {'Voittaja':>10}")
    print("-" * 95)

    for dim in DIMENSIONS:
        wr = win_rates[dim["key"]]
        winner_label = overall_winner[dim["key"]]
        print(
            f"{dim['fi']:<22}"
            f"{wr['A_win_rate']*100:>7.1f}%"
            f"{wr['B_win_rate']*100:>8.1f}%"
            f"{wr['tie_rate']*100:>6.1f}%"
            f"  {wr['A_score_avg']:>8.1f}%"
            f"{wr['B_score_avg']:>9.1f}%"
            f"  {winner_label:>10}"
        )

    print("-" * 95)
    print(
        f"{'YHTEENSÄ (ka)':<22}"
        f"{'':>8}{'':>8}{'':>6}"
        f"  {total_score_a:>8.1f}%"
        f"{total_score_b:>9.1f}%"
        f"  {total_winner_label:>10}"
    )
    print("=" * 95)
    print(f"Kysymyksiä:    {n}")
    print(f"Kesto:         {str(timedelta(seconds=int(duration)))}")
    print(f"Tokeneita:     {total_tokens['input'] + total_tokens['output']:,}")
    print(f"Kustannus:     ${total_cost:.4f}")
    print("=" * 80 + "\n")

    # ── Tallenna JSON ─────────────────────────────────────────────────────
    json_output = {
        "metadata": {
            "method":        "edge_et_al_2024",
            "scoring":       "relative_100pct",
            "timestamp":     timestamp,
            "mode_a":        mode_a,
            "mode_b":        mode_b,
            "judge_model":   JUDGE_MODEL,
            "n_questions":   n,
            "duration_s":    round(duration, 1),
            "tokens":        total_tokens,
            "cost_estimate": {
                "input":  round(input_cost, 6),
                "output": round(output_cost, 6),
                "total":  round(total_cost, 6),
            },
        },
        "dimensions":      [d["key"] for d in DIMENSIONS],
        "win_rates":       win_rates,
        "overall_winner":  overall_winner,
        "total_scores": {
            "A":      total_score_a,
            "B":      total_score_b,
            "sum":    round(total_score_a + total_score_b, 1),
            "winner": total_winner_label,
        },
        "results": results,
    }

    json_path = output_dir / f"er_{mode_a}_vs_{mode_b}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON tallennettu: {json_path}")

    # ── Tallenna Markdown ─────────────────────────────────────────────────
    md_path = output_dir / f"er_{mode_a}_vs_{mode_b}_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:

        f.write(f"# Evaluointiraportti – Edge et al. (2024)\n\n")
        f.write(f"**Menetelmä:** LLM-as-Judge parivertailu (GraphRAG / LightRAG -tutkimuksissa käytetty)\n\n")
        f.write(f"**Pisteytys:** Suhteellinen jako – A-osuus + B-osuus = 100 % (50 % = tasaväkinen)\n\n")
        f.write(f"**Luotu:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Mode A:** `{mode_a}` &nbsp;&nbsp; **Mode B:** `{mode_b}`\n\n")
        f.write(f"**Arvioija-LLM:** `{JUDGE_MODEL}`\n\n")
        f.write(f"**Kysymyksiä:** {n} kpl\n\n")
        f.write("---\n\n")

        # Yhteenveto-taulukko
        f.write("## 📊 Yhteenveto – Win ratet ja suhteelliset pisteet\n\n")
        f.write(f"> Pisteet ovat suhteellisia: A-osuus + B-osuus = 100 %. "
                f"Voittaja saa yli 55 %, tasapeli kun molemmat 45–55 %.\n\n")
        f.write(f"| Dimensio | Mode A ({mode_a}) win% | Mode B ({mode_b}) win% | Tasapeli | Mode A osuus | Mode B osuus | Voittaja |\n")
        f.write(f"|----------|:-------------------:|:-------------------:|:--------:|:------------:|:------------:|----------|\n")

        for dim in DIMENSIONS:
            wr     = win_rates[dim["key"]]
            winner = overall_winner[dim["key"]]
            f.write(
                f"| **{dim['fi']}** "
                f"| {wr['A_win_rate']*100:.1f}% ({wr['A_wins']} voittoa) "
                f"| {wr['B_win_rate']*100:.1f}% ({wr['B_wins']} voittoa) "
                f"| {wr['tie_rate']*100:.1f}% "
                f"| **{wr['A_score_avg']:.1f}%** "
                f"| **{wr['B_score_avg']:.1f}%** "
                f"| **{winner}** |\n"
            )

        f.write(
            f"| **YHTEENSÄ (ka)** | | | "
            f"| **{total_score_a:.1f}%** "
            f"| **{total_score_b:.1f}%** "
            f"| **{total_winner_label}** |\n"
        )

        f.write("\n---\n\n")

        # Kustannukset
        f.write("## 💰 Kustannusarvio\n\n")
        f.write(f"| | Arvo |\n|---|---|\n")
        f.write(f"| Kesto | {str(timedelta(seconds=int(duration)))} |\n")
        f.write(f"| Input tokens | {total_tokens['input']:,} |\n")
        f.write(f"| Output tokens | {total_tokens['output']:,} |\n")
        f.write(f"| **Kokonaiskustannus** | **${total_cost:.4f}** |\n\n")
        f.write("---\n\n")

        # Yksittäiset tulokset
        f.write("## 🔍 Kysymyskohtaiset Tulokset\n\n")

        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            user_desc = meta.get("user_description", "")
            task_desc = meta.get("task_description", "")

            f.write(f"### [{i}] {r['question']}\n\n")

            if user_desc:
                f.write(f"*Käyttäjä:* {user_desc}  \n")
            if task_desc:
                f.write(f"*Tehtävä:* {task_desc}\n\n")

            # Tuomarin päätökset + pisteet
            f.write(f"| Dimensio | A osuus | B osuus | Voittaja | Perustelu |\n")
            f.write(f"|----------|:-------:|:-------:|----------|----------|\n")
            q_scores_a, q_scores_b = [], []
            for dim in DIMENSIONS:
                j      = r["judgments"][dim["key"]]
                winner = j["winner"]
                q_scores_a.append(j["score_a"])
                q_scores_b.append(j["score_b"])
                label  = f"Mode {winner} (`{mode_a if winner == 'A' else mode_b}`)" if winner in ("A", "B") else "Tasapeli"
                reason = j["reasoning"].replace("|", "\\|")
                f.write(f"| {dim['fi']} | {j['score_a']:.1f}% | {j['score_b']:.1f}% | {label} | {reason} |\n")

            q_avg_a = round(sum(q_scores_a) / len(q_scores_a), 1)
            q_avg_b = round(100.0 - q_avg_a, 1)
            q_winner_key = _winner_from_scores(q_avg_a, q_avg_b)
            q_winner = (f"`{mode_a}`" if q_winner_key == "A"
                        else f"`{mode_b}`" if q_winner_key == "B"
                        else "Tasapeli")
            f.write(f"| **Yhteensä (ka)** | **{q_avg_a:.1f}%** | **{q_avg_b:.1f}%** | **{q_winner}** | |\n")

            f.write("\n")
            f.write(f"<details>\n<summary>Näytä vastaukset</summary>\n\n")
            f.write(f"**Mode A (`{mode_a}`):**\n\n{r['answer_a']}\n\n")
            f.write(f"**Mode B (`{mode_b}`):**\n\n{r['answer_b']}\n\n")
            f.write(f"</details>\n\n---\n\n")

    print(f"✓ Markdown tallennettu: {md_path}\n")


# ============================================================================
# KÄYNNISTYS
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Edge et al. (2024) LLM-as-Judge parivertailu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-t", "--testfile",  default=TEST_FILE,  help=f"Testikysymystiedosto (oletus: {TEST_FILE})")
    parser.add_argument("-a", "--mode-a",    default=MODE_A,     help=f"Ensimmäinen mode (oletus: {MODE_A})")
    parser.add_argument("-b", "--mode-b",    default=MODE_B,     help=f"Toinen mode (oletus: {MODE_B})")
    parser.add_argument("-e", "--endpoint",  default=LIGHTRAG_ENDPOINT, help="LightRAG endpoint")
    parser.add_argument("-j", "--judge",     default=JUDGE_MODEL, help=f"Arvioija-LLM (oletus: {JUDGE_MODEL})")

    args = parser.parse_args()

    JUDGE_MODEL       = args.judge
    LIGHTRAG_ENDPOINT = args.endpoint

    main(args.testfile, args.mode_a, args.mode_b)
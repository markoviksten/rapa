#!/usr/bin/env python3
import os, sys, re, json, random, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import requests
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

OPENAI_API_KEY     = os.getenv("LLM_BINDING_API_KEY")
LIGHTRAG_ENDPOINT  = "http://localhost:9623"
LIGHTRAG_JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6NDkyMzgxMzIzMCwicm9sZSI6InVzZXIiLCJtZXRhZGF0YSI6eyJhdXRoX21vZGUiOiJlbmFibGVkIn19.0BTuVkGRO2wljpwiZoKmob22_Z7FiIj0VdqgPsKMHh4"
LIGHTRAG_API_KEY   = "your-secure-api-key-here-marko"
MODE_A = "naive"
MODE_B = "mix"
TEST_FILE = "studio/testcases/tc_20260305_123746.json"
JUDGE_MODEL = "gpt-4o"
TOP_K = 1          # naive-moden chunk-maara
USER_PROMPT = ""
GRAPH_MODE_PARAMS = {
    "top_k":               50,
    "chunk_top_k":         1,
    "max_entity_tokens":   6000,
    "max_relation_tokens": 8000,
    "max_total_tokens":    300000,
    "enable_rerank":       True,
}
RANDOMIZE_AB_ORDER = True
TOKEN_PRICES = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
}

if not OPENAI_API_KEY:
    raise ValueError("LLM_BINDING_API_KEY ei loydy .env-tiedostosta!")
client = OpenAI(api_key=OPENAI_API_KEY)

script_dir   = Path(__file__).parent.absolute()
project_root = script_dir
while project_root.name and not ((project_root/"data").exists() or (project_root/"studio").exists()):
    if project_root.parent == project_root:
        break
    project_root = project_root.parent
if not ((project_root/"data").exists() or (project_root/"studio").exists()):
    project_root = script_dir
output_dir = project_root / "studio" / "testresults"
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

NL = chr(10)

def _raw_to_text(raw) -> str:
    if isinstance(raw, list):
        return NL.join(str(x) for x in raw if x)
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            items = json.loads(raw)
            return NL.join(str(x) for x in items if x)
        except Exception:
            pass
    return raw or ""

def _parse_context_string(ctx_text: str) -> Tuple[int, int, int, List[str]]:
    n_entities = n_relations = n_chunks = 0
    contexts = []
    m = re.search(r"Knowledge Graph Data \(Entity\):(.*?)(?:Knowledge Graph Data \(Relationship\)|Document Chunks|$)", ctx_text, re.DOTALL)
    if m:
        n_entities = sum(1 for l in m.group(1).splitlines() if l.strip().startswith("{"))
    m = re.search(r"Knowledge Graph Data \(Relationship\):(.*?)(?:Document Chunks|$)", ctx_text, re.DOTALL)
    if m:
        n_relations = sum(1 for l in m.group(1).splitlines() if l.strip().startswith("{"))
    m = re.search(r"Document Chunks.*?:(.*?)$", ctx_text, re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                text = obj.get("content", "").strip()
                if text:
                    contexts.append(text)
                    n_chunks += 1
            except Exception:
                pass
    return n_entities, n_relations, n_chunks, contexts

def query_lightrag(question: str, mode: str) -> Tuple[str, List[str], Dict]:
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LIGHTRAG_JWT_TOKEN}",
        "X-API-Key": LIGHTRAG_API_KEY,
    }
    is_naive = mode == "naive"
    base: Dict[str, Any] = {
        "query": question, "mode": mode,
        "top_k":       TOP_K if is_naive else GRAPH_MODE_PARAMS["top_k"],
        "chunk_top_k": TOP_K if is_naive else GRAPH_MODE_PARAMS["chunk_top_k"],
        "only_need_context": False,
        "response_type": "multiple paragraphs",
        "include_references": True, "include_chunk_content": True, "stream": False,
    }
    if not is_naive:
        base.update({
            "chunk_top_k":         GRAPH_MODE_PARAMS["chunk_top_k"],
            "max_entity_tokens":   GRAPH_MODE_PARAMS["max_entity_tokens"],
            "max_relation_tokens": GRAPH_MODE_PARAMS["max_relation_tokens"],
            "max_total_tokens":    GRAPH_MODE_PARAMS["max_total_tokens"],
            "enable_rerank":       GRAPH_MODE_PARAMS["enable_rerank"],
        })
    if USER_PROMPT:
        base["user_prompt"] = USER_PROMPT
    try:
        r = requests.post(f"{LIGHTRAG_ENDPOINT}/query", json={**base, "only_need_context": False}, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        answer = data.get("response", "")
        refs = data.get("references", [])
        contexts, n_chunks = [], 0
        for ref in refs:
            if isinstance(ref, str):
                text = ref.strip()
            elif isinstance(ref, dict):
                text = _raw_to_text(ref.get("content", "")).strip()
            else:
                continue
            if text:
                contexts.append(text)
                n_chunks += 1
        n_entities = n_relations = 0
        if not is_naive:
            try:
                r2 = requests.post(f"{LIGHTRAG_ENDPOINT}/query", json={**base, "only_need_context": True}, headers=headers, timeout=60)
                r2.raise_for_status()
                ctx_data = r2.json()
                ctx_text = _raw_to_text(ctx_data.get("response", "") or ctx_data.get("context", ""))
                n_ent, n_rel, n_chk, ctx_ctx = _parse_context_string(ctx_text)
                n_entities, n_relations = n_ent, n_rel
                if n_chk > n_chunks:
                    n_chunks, contexts = n_chk, ctx_ctx
            except Exception as e:
                print(f"    stats-kutsu epaonnistui ({mode}): {e}")
        return answer, contexts, {"chunks": n_chunks, "entities": n_entities, "relations": n_relations, "words": len(answer.split())}
    except Exception as e:
        print(f"    LightRAG-virhe ({mode}): {e}")
        return "", [], {"chunks": 0, "entities": 0, "relations": 0, "words": 0}

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing the quality of AI-generated answers. "
    "Compare two answers (Answer A and Answer B) to the same question. "
    "Be objective and base your judgment solely on the content. "
    "Always respond with valid JSON only."
)

def build_judge_prompt(question, answer_a, answer_b, dimension, dimension_fi, dimension_desc):
    return (
        f"Question: {question}" + NL + NL +
        f"Answer A:" + NL + f"{answer_a}" + NL + NL +
        f"Answer B:" + NL + f"{answer_b}" + NL + NL +
        f"Evaluate based on **{dimension} ({dimension_fi})**: {dimension_desc}" + NL + NL +
        "## Scoring" + NL +
        "Distribute 100 points between the answers to reflect RELATIVE quality. Scores must sum to 100.00." + NL +
        "Use the FULL scale - do not cluster around 50/50:" + NL +
        "  50/50 = truly equal, no meaningful difference" + NL +
        "  60/40 = one answer is noticeably better" + NL +
        "  70/30 = one answer is clearly better" + NL +
        "  80/20 = one answer is much better" + NL +
        "  90/10 = one answer is far superior" + NL + NL +
        "## Depth of Understanding Bonus (most important factor)" + NL +
        "Strongly reward answers that demonstrate BROADER and DEEPER understanding:" + NL +
        "  +++ Traces causal chains and explains WHY things happen, not just WHAT" + NL +
        "  +++ Connects multiple entities/concepts to each other (e.g. A affects B which leads to C)" + NL +
        "  +++ Reveals non-obvious relationships or second-order effects" + NL +
        "  +++ Synthesizes information across different domains or levels (tactical/operational/strategic)" + NL +
        "  +++ Provides context that helps understand the broader significance" + NL + NL +
        "Penalize answers that:" + NL +
        "  --- Only list facts without explaining connections" + NL +
        "  --- Stay at surface level without showing how concepts relate" + NL +
        "  --- Miss the systemic or relational dimension of the question" + NL + NL +
        "A score of 55/45 should be RARE - only when answers are nearly identical in depth." + NL +
        'Winner: "A" if score_a>53, "B" if score_b>53, "tie" only if genuinely indistinguishable.' + NL + NL +
        'Respond ONLY with: {"score_a": <float>, "score_b": <float>, "winner": "A"|"B"|"tie", "reasoning": "Finnish 1-2 sentences: what specific deeper understanding did the better answer show?"}'
    )

DIMENSIONS = [
    {"key": "kattavuus",      "en": "Comprehensiveness", "fi": "Kattavuus",      "desc": "How thoroughly does the answer address all aspects?"},
    {"key": "monipuolisuus",  "en": "Diversity",          "fi": "Monipuolisuus",  "desc": "How diverse are the perspectives and information presented?"},
    {"key": "oivalluttavuus", "en": "Empowerment",        "fi": "Oivalluttavuus", "desc": "How well does the answer help the reader understand and form conclusions?"},
]

def _normalize(a, b):
    t = a + b
    if t <= 0: return 50.0, 50.0
    na = round(a * 100.0 / t, 2)
    return na, round(100.0 - na, 2)

def _winner(sa, sb):
    if sa > 53.0: return "A"
    if sb > 53.0: return "B"
    return "tie"

def judge_pair(question, answer_a, answer_b, swapped):
    judgments = {}
    tokens = {"input": 0, "output": 0}
    for dim in DIMENSIONS:
        prompt = build_judge_prompt(question, answer_a, answer_b, dim["en"], dim["fi"], dim["desc"])
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                max_tokens=250, temperature=0, response_format={"type": "json_object"},
            )
            tokens["input"]  += resp.usage.prompt_tokens
            tokens["output"] += resp.usage.completion_tokens
            j = json.loads(resp.choices[0].message.content)
            na, nb = _normalize(float(j.get("score_a", 50.0)), float(j.get("score_b", 50.0)))
            sa, sb = (nb, na) if swapped else (na, nb)
            judgments[dim["key"]] = {"winner": _winner(sa, sb), "score_a": sa, "score_b": sb, "reasoning": j.get("reasoning", "")}
        except Exception as e:
            print(f"      Arviointivirhe ({dim['fi']}): {e}")
            judgments[dim["key"]] = {"winner": "tie", "score_a": 50.0, "score_b": 50.0, "reasoning": f"Virhe: {e}"}
        time.sleep(0.3)
    return judgments, tokens

def main(test_file, mode_a, mode_b):
    SEP80 = "=" * 80
    print(SEP80)
    print("EDGE ET AL. (2024) - LLM-AS-JUDGE PARIVERTAILU")
    print(SEP80)
    print(f"Mode A:   {mode_a}")
    print(f"Mode B:   {mode_b}")
    print(f"Arvioija: {JUDGE_MODEL}")
    print(f"Testit:   {test_file}")
    print(SEP80 + NL)

    fp = project_root / test_file
    if not fp.exists(): fp = Path(test_file)
    if not fp.exists():
        print(f"Tiedostoa ei loydy: {test_file}")
        sys.exit(1)
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_cases = data.get("test_cases", [])
    if not test_cases:
        print("Ei testikysymyksia.")
        sys.exit(1)
    print(f"Ladattu {len(test_cases)} kysymysta" + NL)

    results = []
    total_tokens = {"input": 0, "output": 0}
    start = time.time()
    wins   = {d["key"]: {"A": 0, "B": 0, "tie": 0} for d in DIMENSIONS}
    scores = {d["key"]: {"A": [], "B": []} for d in DIMENSIONS}

    for idx, tc in enumerate(test_cases, 1):
        question = tc.get("question", "")
        meta     = tc.get("metadata", {})
        print(f"[{idx}/{len(test_cases)}] {question[:80]}...")
        print(f"  -> Haetaan mode={mode_a}...")
        ans_a, ctx_a, stats_a = query_lightrag(question, mode_a)
        print(f"  -> Haetaan mode={mode_b}...")
        ans_b, ctx_b, stats_b = query_lightrag(question, mode_b)
        swapped = RANDOMIZE_AB_ORDER and random.random() < 0.5
        judge_a = ans_b if swapped else ans_a
        judge_b = ans_a if swapped else ans_b
        def _pct(a, b):
            if a + b == 0: return "  -"
            return f"{a/(a+b)*100:4.0f}%"
        ea, eb = stats_a['entities'], stats_b['entities']
        ra, rb = stats_a['relations'], stats_b['relations']
        ca, cb = stats_a['chunks'], stats_b['chunks']
        wa, wb = stats_a['words'], stats_b['words']
        print(f"     {'':10s}  {'chunkit':>8}  {'entit.':>8}  {'relaat.':>8}  {'sanat':>6}")
        print(f"     {mode_a:10s}: {ca:>6}     {ea:>6}     {ra:>6}     {wa:>6}")
        print(f"     {mode_b:10s}: {cb:>6}     {eb:>6}     {rb:>6}     {wb:>6}")
        print(f"     {'mix-etu':10s}: {_pct(cb-ca,ca):>6}    {_pct(eb,0):>6}    {_pct(rb,0):>6}    {_pct(wb-wa,wa):>6}")
        print(f"  -> Arvioidaan ({JUDGE_MODEL}){('  [vaihdettu]' if swapped else '')}...")
        judgments, qtok = judge_pair(question, judge_a, judge_b, swapped)
        total_tokens["input"]  += qtok["input"]
        total_tokens["output"] += qtok["output"]
        qa, qb = 0.0, 0.0
        for dk, j in judgments.items():
            wins[dk][j["winner"]] = wins[dk].get(j["winner"], 0) + 1
            scores[dk]["A"].append(j["score_a"])
            scores[dk]["B"].append(j["score_b"])
            qa += j["score_a"]; qb += j["score_b"]
        results.append({"question": question, "metadata": meta, "answer_a": ans_a, "answer_b": ans_b,
                        "contexts_a": ctx_a, "contexts_b": ctx_b, "stats_a": stats_a, "stats_b": stats_b,
                        "judgments": judgments, "swapped": swapped})
        for dim in DIMENSIONS:
            j = judgments[dim["key"]]
            lbl = f"Mode {j['winner']}" if j["winner"] in ("A","B") else "Tasapeli"
            print(f"     {dim['fi']:20s}: {lbl:10s}  A={j['score_a']:5.1f}%  B={j['score_b']:5.1f}%")
        print(f"     {'Kokonais':20s}:            A={qa/len(DIMENSIONS):5.1f}%  B={qb/len(DIMENSIONS):5.1f}%" + NL)
        time.sleep(0.5)

    duration = time.time() - start
    n = len(test_cases)
    win_rates = {}
    for dk, w in wins.items():
        sa, sb = scores[dk]["A"], scores[dk]["B"]
        win_rates[dk] = {
            "A_wins": w.get("A",0), "B_wins": w.get("B",0), "ties": w.get("tie",0),
            "A_win_rate": round(w.get("A",0)/n,4), "B_win_rate": round(w.get("B",0)/n,4), "tie_rate": round(w.get("tie",0)/n,4),
            "A_score_avg": round(sum(sa)/len(sa),1) if sa else 50.0,
            "B_score_avg": round(sum(sb)/len(sb),1) if sb else 50.0,
        }
    all_a = [s for d in DIMENSIONS for s in scores[d["key"]]["A"]]
    all_b = [s for d in DIMENSIONS for s in scores[d["key"]]["B"]]
    total_a = round(sum(all_a)/len(all_a),1) if all_a else 50.0
    total_b = round(100.0-total_a,1)
    ow = {}
    for dk, wr in win_rates.items():
        ow[dk] = mode_a if wr["A_wins"]>wr["B_wins"] else mode_b if wr["B_wins"]>wr["A_wins"] else "tasapeli"
    tw_key = _winner(total_a, total_b)
    tw_lbl = mode_a if tw_key=="A" else mode_b if tw_key=="B" else "tasapeli"
    prices = TOKEN_PRICES.get(JUDGE_MODEL, {"input":0,"output":0})
    ic = (total_tokens["input"]/1e6)*prices["input"]
    oc = (total_tokens["output"]/1e6)*prices["output"]
    tc_ = ic+oc

    SEP95 = "=" * 95
    print(SEP95)
    print("TULOKSET  (A + B = 100 %)")
    print(SEP95)
    print(f"{'Dimensio':<22} {'A win%':>8} {'B win%':>8} {'Tas%':>6}  {'A osuus':>9} {'B osuus':>9}  {'Voittaja':>10}")
    print("-" * 95)
    for dim in DIMENSIONS:
        wr = win_rates[dim["key"]]
        print(f"{dim['fi']:<22}{wr['A_win_rate']*100:>7.1f}%{wr['B_win_rate']*100:>8.1f}%{wr['tie_rate']*100:>6.1f}%  {wr['A_score_avg']:>8.1f}%{wr['B_score_avg']:>9.1f}%  {ow[dim['key']]:>10}")
    print("-" * 95)
    print(f"{'YHTEENSA (ka)':<22}{'':>8}{'':>8}{'':>6}  {total_a:>8.1f}%{total_b:>9.1f}%  {tw_lbl:>10}")
    print(SEP95)
    print(f"Kysymyksia: {n}  |  Kesto: {str(timedelta(seconds=int(duration)))}  |  Tokeneita: {total_tokens['input']+total_tokens['output']:,}  |  Kustannus: ${tc_:.4f}" + NL)

    jout = {
        "metadata": {"method":"edge_et_al_2024","scoring":"relative_100pct","timestamp":timestamp,
                     "mode_a":mode_a,"mode_b":mode_b,"judge_model":JUDGE_MODEL,"n_questions":n,
                     "duration_s":round(duration,1),"tokens":total_tokens,
                     "cost_estimate":{"input":round(ic,6),"output":round(oc,6),"total":round(tc_,6)}},
        "dimensions":[d["key"] for d in DIMENSIONS],"win_rates":win_rates,"overall_winner":ow,
        "total_scores":{"A":total_a,"B":total_b,"sum":round(total_a+total_b,1),"winner":tw_lbl},
        "results":results,
    }
    jp = output_dir/f"er_{mode_a}_vs_{mode_b}_{timestamp}.json"
    with open(jp,"w",encoding="utf-8") as f: json.dump(jout,f,ensure_ascii=False,indent=2)
    print(f"JSON: {jp}")

    mp = output_dir/f"er_{mode_a}_vs_{mode_b}_{timestamp}.md"
    with open(mp,"w",encoding="utf-8") as f:
        f.write(f"# Evaluointiraportti - Edge et al. (2024)" + NL + NL)
        f.write(f"**Mode A:** `{mode_a}` | **Mode B:** `{mode_b}` | **Arvioija:** `{JUDGE_MODEL}` | **Kysymyksia:** {n}" + NL + NL)
        f.write("---" + NL + NL + "## Yhteenveto" + NL + NL)
        f.write(f"| Dimensio | {mode_a} win% | {mode_b} win% | Tas% | {mode_a} osuus | {mode_b} osuus | Voittaja |" + NL)
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|" + NL)
        for dim in DIMENSIONS:
            wr = win_rates[dim["key"]]
            f.write(f"| **{dim['fi']}** | {wr['A_win_rate']*100:.1f}% ({wr['A_wins']}) | {wr['B_win_rate']*100:.1f}% ({wr['B_wins']}) | {wr['tie_rate']*100:.1f}% | **{wr['A_score_avg']:.1f}%** | **{wr['B_score_avg']:.1f}%** | **{ow[dim['key']]}** |" + NL)
        f.write(f"| **YHTEENSA** | | | | **{total_a:.1f}%** | **{total_b:.1f}%** | **{tw_lbl}** |" + NL + NL)
        f.write("---" + NL + NL + "## Kysymyskohtaiset tulokset" + NL + NL)
        for i, r in enumerate(results, 1):
            f.write(f"### [{i}] {r['question']}" + NL + NL)
            sa, sb = r.get("stats_a",{}), r.get("stats_b",{})
            f.write(f"| | Chunkit | Entiteetit | Relaatiot | Sanat |" + NL + "|---|:---:|:---:|:---:|:---:|" + NL)
            f.write(f"| **{mode_a}** | {sa.get('chunks',0)} | {sa.get('entities',0)} | {sa.get('relations',0)} | {sa.get('words',0)} |" + NL)
            f.write(f"| **{mode_b}** | {sb.get('chunks',0)} | {sb.get('entities',0)} | {sb.get('relations',0)} | {sb.get('words',0)} |" + NL + NL)
            f.write("| Dimensio | A osuus | B osuus | Voittaja | Perustelu |" + NL + "|---|:---:|:---:|---|---|" + NL)
            qa, qb = [], []
            for dim in DIMENSIONS:
                j = r["judgments"][dim["key"]]
                qa.append(j["score_a"]); qb.append(j["score_b"])
                w = j["winner"]
                lbl = f"Mode {w} (`{mode_a if w=='A' else mode_b}`)" if w in ("A","B") else "Tasapeli"
                f.write(f"| {dim['fi']} | {j['score_a']:.1f}% | {j['score_b']:.1f}% | {lbl} | {j['reasoning'].replace('|','|')} |" + NL)
            avg_a = round(sum(qa)/len(qa),1); avg_b = round(100.0-avg_a,1)
            wk = _winner(avg_a,avg_b)
            wl = f"`{mode_a}`" if wk=="A" else f"`{mode_b}`" if wk=="B" else "Tasapeli"
            f.write(f"| **Yhteensa** | **{avg_a:.1f}%** | **{avg_b:.1f}%** | **{wl}** | |" + NL + NL)
            f.write(f"<details><summary>Nayta vastaukset</summary>" + NL + NL)
            f.write(f"**{mode_a}:**" + NL + NL + r['answer_a'] + NL + NL)
            f.write(f"**{mode_b}:**" + NL + NL + r['answer_b'] + NL + NL)
            f.write("</details>" + NL + NL + "---" + NL + NL)
    print(f"Markdown: {mp}" + NL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testfile", default=TEST_FILE)
    parser.add_argument("-a", "--mode-a",   default=MODE_A)
    parser.add_argument("-b", "--mode-b",   default=MODE_B)
    parser.add_argument("-e", "--endpoint", default=LIGHTRAG_ENDPOINT)
    parser.add_argument("-j", "--judge",    default=JUDGE_MODEL)
    args = parser.parse_args()
    JUDGE_MODEL = args.judge
    LIGHTRAG_ENDPOINT = args.endpoint
    main(args.testfile, args.mode_a, args.mode_b)
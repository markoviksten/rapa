import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import time
from datetime import datetime, timedelta
import random

load_dotenv()

api_key = os.getenv("LLM_BINDING_API_KEY")
if not api_key:
    raise ValueError("LLM_BINDING_API_KEY ei löydy .env-tiedostosta!")

client = OpenAI(api_key=api_key)

# ============================================================================
# KONFIGURAATIO
# ============================================================================

# Malli
MODEL = "gpt-4o-mini"

# Datahakemisto
DATA_DIR = "data/nano3/rag_storage"

# Tiedostot – script yrittää löytää nämä automaattisesti
CHUNKS_FILE   = "vdb_chunks.json"
GRAPHML_FILE  = "graph_chunk_entity_relation.graphml"
ENTITIES_FILE = "vdb_entities.json"       # vaihtoehto jos GraphML puuttuu
RELATIONS_FILE = "vdb_relationships.json"  # vaihtoehto jos GraphML puuttuu

# Kuinka monta chunkkia / entiteettiä / relaatiota otetaan mukaan kuvaukseen
DESCRIPTION_CHUNK_LIMIT    = 30
GRAPH_ENTITY_SAMPLE        = 80   # entiteettiä dataset-kuvaukseen
GRAPH_RELATION_SAMPLE      = 80   # relaatiota dataset-kuvaukseen

# Käyttäjien ja tehtävien määrä (yhteinen molemmille)
NUM_USERS = 5
NUM_TASKS = 5

# Kysymyksiä per (käyttäjä, tehtävä) – erikseen graph ja chunk
# Graph-kysymykset testaavat mix/local/global -modeja
# Chunk-kysymykset toimivat baseline-vertailuna (naive)
NUM_QUESTIONS_GRAPH = 5   # → yhteensä NUM_USERS * NUM_TASKS * tämä graph-kysymystä
NUM_QUESTIONS_CHUNK = 0   # → yhteensä NUM_USERS * NUM_TASKS * tämä chunk-kysymystä

# Kieli
LANGUAGE = "Finnish"

# Token-hinnat
TOKEN_PRICES = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
}

# ============================================================================
# Älä muokkaa tästä alaspäin
# ============================================================================

script_dir   = Path(__file__).parent.absolute()
project_root = script_dir
while project_root.name and not (project_root / "data").exists():
    if project_root.parent == project_root:
        break
    project_root = project_root.parent
if not (project_root / "data").exists():
    project_root = script_dir

output_dir = project_root / "studio" / "testcases"
output_dir.mkdir(parents=True, exist_ok=True)
os.chdir(output_dir)

timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = Path(f"tc_{timestamp}.json")
MD_FILE     = Path(f"tcr_{timestamp}.md")

print(f"Projektin juuri:  {project_root}")
print(f"Output hakemisto: {output_dir}")
print(f"Output tiedosto:  {OUTPUT_FILE}\n")
print("=" * 80)
print("EDGE ET AL. (2024) – TESTIKYSYMYSTEN GENEROINTI (CHUNKS + GRAPH)")
print("=" * 80)

# ============================================================================
# APUFUNKTIOT – DATAN LUKU
# ============================================================================

def load_chunks(data_dir: Path) -> list:
    """Lataa vektori-chunkit."""
    path = data_dir / CHUNKS_FILE
    if not path.exists():
        print(f"  ⚠️  Chunk-tiedostoa ei löydy: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("data", data) if isinstance(data, dict) else data
    print(f"  ✓ Chunkit: {len(chunks)} kpl ({path.name})")
    return chunks


def load_graph_graphml(data_dir: Path) -> tuple[list, list]:
    """Lataa entiteetit ja relaatiot GraphML-tiedostosta."""
    path = data_dir / GRAPHML_FILE
    if not path.exists():
        print(f"  ⚠️  GraphML-tiedostoa ei löydy: {path}")
        return [], []

    print(f"  → Löytyi: {path} ({path.stat().st_size / 1024:.1f} KB)")

    try:
        tree = ET.parse(path)
        root = tree.getroot()

        # Tunnista namespace automaattisesti root-tagista
        # esim. "{http://graphml.graphdrawing.org/graphml}graphml" → ns = "http://..."
        raw_tag = root.tag  # esim. "{http://graphml.graphdrawing.org/graphml}graphml"
        if raw_tag.startswith("{"):
            detected_ns = raw_tag[1:raw_tag.index("}")]
            ns = {"g": detected_ns}
            print(f"  → Namespace: {detected_ns}")
        else:
            ns = None
            print(f"  → Ei namespacea, käytetään suoraa tagihakua")

        def find_all(parent, tag):
            if ns:
                return parent.findall(f"g:{tag}", ns)
            return parent.findall(tag)

        def find_one(parent, tag):
            if ns:
                return parent.find(f"g:{tag}", ns)
            return parent.find(tag)

        # Kerää key-määritykset (id → attr.name)
        keys = {}
        for key in find_all(root, "key"):
            keys[key.get("id")] = key.get("attr.name", key.get("id"))
        print(f"  → Key-määritykset: {list(keys.values())[:10]}")

        entities  = []
        relations = []

        graph_el = find_one(root, "graph")
        if graph_el is None:
            print(f"  ⚠️  <graph>-elementtiä ei löydy GraphML:stä!")
            print(f"  → Root-tagin lapset: {[c.tag for c in root][:10]}")
            return [], []

        for node in find_all(graph_el, "node"):
            entity = {"id": node.get("id")}
            for data_el in find_all(node, "data"):
                attr = keys.get(data_el.get("key"), data_el.get("key"))
                entity[attr] = (data_el.text or "").strip()
            entities.append(entity)

        for edge in find_all(graph_el, "edge"):
            rel = {
                "source": edge.get("source"),
                "target": edge.get("target"),
            }
            for data_el in find_all(edge, "data"):
                attr = keys.get(data_el.get("key"), data_el.get("key"))
                rel[attr] = (data_el.text or "").strip()
            relations.append(rel)

        print(f"  ✓ GraphML: {len(entities)} entiteettiä, {len(relations)} relaatiota")
        return entities, relations

    except ET.ParseError as e:
        print(f"  ❌ GraphML XML-parsausvirhe: {e}")
        return [], []
    except Exception as e:
        print(f"  ❌ GraphML-virhe: {e}")
        return [], []


def load_graph_json(data_dir: Path) -> tuple[list, list]:
    """Vaihtoehto: lataa entiteetit/relaatiot JSON-tiedostoista."""
    entities  = []
    relations = []

    ep = data_dir / ENTITIES_FILE
    if ep.exists():
        with open(ep, "r", encoding="utf-8") as f:
            d = json.load(f)
        entities = d.get("data", d) if isinstance(d, dict) else d
        print(f"  ✓ Entiteetit (JSON): {len(entities)} kpl")

    rp = data_dir / RELATIONS_FILE
    if rp.exists():
        with open(rp, "r", encoding="utf-8") as f:
            d = json.load(f)
        relations = d.get("data", d) if isinstance(d, dict) else d
        print(f"  ✓ Relaatiot (JSON):  {len(relations)} kpl")

    return entities, relations


def format_entities_for_prompt(entities: list, sample_size: int) -> str:
    """Muodosta entiteettilista promptia varten."""
    sample = random.sample(entities, min(sample_size, len(entities)))
    lines  = []
    for e in sample:
        name  = e.get("entity_id") or e.get("id") or e.get("name") or ""
        etype = e.get("entity_type") or e.get("type") or ""
        desc  = e.get("description") or e.get("desc") or ""
        if desc and len(desc) > 120:
            desc = desc[:120] + "..."
        line  = f"- [{etype}] {name}"
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines)


def format_relations_for_prompt(relations: list, sample_size: int) -> str:
    """Muodosta relaatiolista promptia varten."""
    sample = random.sample(relations, min(sample_size, len(relations)))
    lines  = []
    for r in sample:
        src  = r.get("source") or r.get("src") or ""
        tgt  = r.get("target") or r.get("tgt") or r.get("dst") or ""
        desc = r.get("description") or r.get("keywords") or r.get("relation") or ""
        if desc and len(desc) > 120:
            desc = desc[:120] + "..."
        line = f"- {src} → {tgt}"
        if desc:
            line += f" ({desc})"
        lines.append(line)
    return "\n".join(lines)


# ============================================================================
# VAIHE 1: Lataa data
# ============================================================================

print("\nVaihe 1/4: Ladataan data...")
data_dir = project_root / DATA_DIR

chunks             = load_chunks(data_dir)
entities, relations = load_graph_graphml(data_dir)

# Fallback JSON:iin jos GraphML ei löydy tai on tyhjä
if not entities and not relations:
    print("  → GraphML ei löytynyt tai tyhjä, yritetään JSON-tiedostoja...")
    entities, relations = load_graph_json(data_dir)

has_graph = bool(entities or relations)
if not has_graph:
    print("  ⚠️  Graafidataa ei löydy – generoidaan vain chunk-pohjaisia kysymyksiä")

# ============================================================================
# VAIHE 2: Dataset-kuvaus
# ============================================================================

print("\nVaihe 2/4: Muodostetaan dataset-kuvaus...")

# Chunk-teksti
sample_chunks = chunks[:DESCRIPTION_CHUNK_LIMIT] if chunks else []
chunk_text    = "\n\n---\n\n".join(
    c.get("content", c) if isinstance(c, dict) else str(c)
    for c in sample_chunks
)

# Entiteetti- ja relaatioteksti
entity_text   = format_entities_for_prompt(entities, GRAPH_ENTITY_SAMPLE)   if entities   else ""
relation_text = format_relations_for_prompt(relations, GRAPH_RELATION_SAMPLE) if relations else ""

description_prompt = f"""Analyze the following dataset excerpts and knowledge graph data, 
then write a concise description (4-6 sentences) of:
- What domain/topic this dataset covers
- What kinds of entities and relationships exist in the knowledge graph
- What the data would typically be used for
- What kinds of analytical questions it could answer

{"DOCUMENT EXCERPTS:" + chr(10) + chunk_text[:6000] if chunk_text else ""}

{"KNOWLEDGE GRAPH - ENTITIES (sample):" + chr(10) + entity_text if entity_text else ""}

{"KNOWLEDGE GRAPH - RELATIONSHIPS (sample):" + chr(10) + relation_text if relation_text else ""}

Write the description in English."""

try:
    desc_response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": description_prompt}],
        max_tokens=600,
        temperature=0.3,
    )
    dataset_description = desc_response.choices[0].message.content.strip()
    desc_tokens = {
        "input":  desc_response.usage.prompt_tokens,
        "output": desc_response.usage.completion_tokens,
        "total":  desc_response.usage.total_tokens,
    }
    print(f"✓ Dataset-kuvaus luotu ({desc_tokens['total']} tokenia)")
    print(f"\n  Kuvaus: {dataset_description[:300]}...\n")
except Exception as e:
    print(f"❌ Virhe dataset-kuvauksessa: {e}")
    exit()

# ============================================================================
# VAIHE 3a: Generoi GRAPH-kysymykset (testaavat mix/local/global)
# ============================================================================

all_test_cases  = []
all_tokens      = desc_tokens.copy()
generation_time = 0.0
graph_structured_data = None
chunk_structured_data = None

if has_graph:
    print("Vaihe 3a/4: Generoidaan GRAPH-kysymykset (mix/local/global -modeille)...")

    # Rakenna kattava näyte graafista promptia varten
    entity_sample_prompt   = format_entities_for_prompt(entities, 60)
    relation_sample_prompt = format_relations_for_prompt(relations, 60)

    graph_master_prompt = f"""You are designing evaluation questions for a Retrieval-Augmented Generation (RAG) system 
that uses a knowledge graph. The goal is to create questions that specifically REQUIRE 
graph-based reasoning — questions that a simple keyword/chunk search CANNOT answer well, 
but a graph-aware system (with entity relationships) CAN.

DATASET DESCRIPTION:
{dataset_description}

KNOWLEDGE GRAPH – ENTITIES (sample of {min(60, len(entities))} out of {len(entities)}):
{entity_sample_prompt}

KNOWLEDGE GRAPH – RELATIONSHIPS (sample of {min(60, len(relations))} out of {len(relations)}):
{relation_sample_prompt}

Please identify {NUM_USERS} potential users who would engage with this dataset.
For each user, list {NUM_TASKS} tasks they would perform.
For each (user, task) combination, generate {NUM_QUESTIONS_GRAPH} questions in {LANGUAGE} that:

REQUIRED question types (mix across all questions):
1. RELATIONSHIP questions: "Miten X liittyy Y:hyn?", "Mikä yhteys on A:n ja B:n välillä?"
2. MULTI-HOP questions: require chaining 2-3 relationships, e.g. "Ketkä ovat Z:n yhteistyökumppaneita jotka myös tekevät A:ta?"
3. GLOBAL SUMMARY questions: require synthesizing patterns across many entities, e.g. "Mitkä ovat yleisimmät teemat X:n ympärillä?"
4. ENTITY-CENTRIC questions: focus on a specific named entity from the graph and its connections
5. COMPARATIVE questions: "Miten A eroaa B:stä X:n suhteen?"

IMPORTANT:
- Use ACTUAL entity names from the knowledge graph above where possible
- Questions must require understanding connections, NOT just finding a text passage
- Questions in {LANGUAGE}
- Open-ended, no simple yes/no answers

Return ONLY a JSON object:
{{
  "users": [
    {{
      "user_id": 1,
      "description": "User description",
      "tasks": [
        {{
          "task_id": 1,
          "description": "Task description",
          "questions": ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"]
        }}
      ]
    }}
  ]
}}"""

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": graph_master_prompt}],
            max_tokens=8000,
            temperature=0.9,
            response_format={"type": "json_object"},
        )
        generation_time += time.time() - t0

        tok = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens, "total": resp.usage.total_tokens}
        all_tokens["input"]  += tok["input"]
        all_tokens["output"] += tok["output"]
        all_tokens["total"]  += tok["total"]

        graph_structured_data = json.loads(resp.choices[0].message.content)

        for user in graph_structured_data.get("users", []):
            for task in user.get("tasks", []):
                for q_idx, question in enumerate(task.get("questions", []), 1):
                    all_test_cases.append({
                        "question": question,
                        "metadata": {
                            "user_id":          user["user_id"],
                            "user_description": user.get("description", ""),
                            "task_id":          task["task_id"],
                            "task_description": task.get("description", ""),
                            "question_index":   q_idx,
                            "question_type":    "graph",   # ← tärkeä: testataan graph-modeja
                            "method":           "edge_et_al_2024",
                        }
                    })

        print(f"  ✓ Graph-kysymykset: {len([tc for tc in all_test_cases if tc['metadata']['question_type'] == 'graph'])} kpl ({tok['total']} tokenia)")

    except Exception as e:
        print(f"  ❌ Virhe graph-kysymyksissä: {e}")

# ============================================================================
# VAIHE 3b: Generoi CHUNK-kysymykset (baseline, toimii naivelle)
# ============================================================================

print("Vaihe 3b/4: Generoidaan CHUNK-kysymykset (baseline / naive-modelle)...")

chunk_master_prompt = f"""Given the following description of a dataset:

{dataset_description}

Please identify {NUM_USERS} potential users who would engage with this dataset.
For each user, list {NUM_TASKS} tasks they would perform with this dataset.
For each (user, task) combination, generate {NUM_QUESTIONS_CHUNK} questions in {LANGUAGE} that:
- Can be answered from specific text passages in the dataset
- Require synthesizing information across multiple sections
- Are open-ended and meaningful (no simple yes/no answers)
- Reflect the specific user's perspective and task context

Return ONLY a JSON object:
{{
  "users": [
    {{
      "user_id": 1,
      "description": "User description",
      "tasks": [
        {{
          "task_id": 1,
          "description": "Task description",
          "questions": ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"]
        }}
      ]
    }}
  ]
}}"""

try:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": chunk_master_prompt}],
        max_tokens=6000,
        temperature=0.8,
        response_format={"type": "json_object"},
    )
    generation_time += time.time() - t0

    tok = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens, "total": resp.usage.total_tokens}
    all_tokens["input"]  += tok["input"]
    all_tokens["output"] += tok["output"]
    all_tokens["total"]  += tok["total"]

    chunk_structured_data = json.loads(resp.choices[0].message.content)

    for user in chunk_structured_data.get("users", []):
        for task in user.get("tasks", []):
            for q_idx, question in enumerate(task.get("questions", []), 1):
                all_test_cases.append({
                    "question": question,
                    "metadata": {
                        "user_id":          user["user_id"],
                        "user_description": user.get("description", ""),
                        "task_id":          task["task_id"],
                        "task_description": task.get("description", ""),
                        "question_index":   q_idx,
                        "question_type":    "chunk",   # ← baseline
                        "method":           "edge_et_al_2024",
                    }
                })

    print(f"  ✓ Chunk-kysymykset: {len([tc for tc in all_test_cases if tc['metadata']['question_type'] == 'chunk'])} kpl ({tok['total']} tokenia)")

except Exception as e:
    print(f"  ❌ Virhe chunk-kysymyksissä: {e}")

# Sekoita järjestys jotta graph/chunk-kysymykset vuorottelevat
random.shuffle(all_test_cases)
total_questions = len(all_test_cases)
graph_count = len([tc for tc in all_test_cases if tc["metadata"]["question_type"] == "graph"])
chunk_count = len([tc for tc in all_test_cases if tc["metadata"]["question_type"] == "chunk"])

print(f"\n✓ Yhteensä {total_questions} kysymystä ({graph_count} graph + {chunk_count} chunk)\n")

# ============================================================================
# KUSTANNUKSET
# ============================================================================

prices      = TOKEN_PRICES.get(MODEL, {"input": 0, "output": 0})
input_cost  = (all_tokens["input"]  / 1_000_000) * prices["input"]
output_cost = (all_tokens["output"] / 1_000_000) * prices["output"]
total_cost  = input_cost + output_cost

# ============================================================================
# TALLENNUS – JSON
# ============================================================================

final_output = {
    "test_cases": all_test_cases,
    "metadata": {
        "method":                    "edge_et_al_2024_graphrag",
        "dataset_description":       dataset_description,
        "total_questions":           total_questions,
        "graph_questions":           graph_count,
        "chunk_questions":           chunk_count,
        "num_users":                 NUM_USERS,
        "num_tasks_per_user":        NUM_TASKS,
        "num_questions_per_task":    NUM_QUESTIONS_GRAPH, "num_chunk_questions_per_task": NUM_QUESTIONS_CHUNK,
        "has_graph_data":            has_graph,
        "graph_stats": {
            "total_entities":  len(entities),
            "total_relations": len(relations),
        },
        "generation_time_seconds":   round(generation_time, 1),
        "generation_time_formatted": str(timedelta(seconds=int(generation_time))),
        "tokens_used":               all_tokens,
        "cost_estimate": {
            "input_cost":  round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost":  round(total_cost, 6),
            "model":       MODEL,
        },
        "graph_structure":  graph_structured_data,
        "chunk_structure":  chunk_structured_data,
    }
}

output_path = OUTPUT_FILE.absolute()
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)
print(f"✓ JSON tallennettu: {output_path}")

# ============================================================================
# TALLENNUS – MARKDOWN
# ============================================================================

md_path = MD_FILE.absolute()
with open(MD_FILE, "w", encoding="utf-8") as f:

    f.write("# Testikysymysten Generointiraportti – Edge et al. (2024)\n\n")
    f.write(f"**Luotu:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Menetelmä:** Edge et al. (2024) – GraphRAG / LLM-as-a-Judge parivertailu\n\n")
    f.write(f"**Kysymystyypit:** Graph-pohjainen ({graph_count} kpl) + Chunk-pohjainen ({chunk_count} kpl)\n\n")
    f.write("---\n\n")

    f.write("## 📄 Dataset-kuvaus\n\n")
    f.write(f"{dataset_description}\n\n")

    if has_graph:
        f.write(f"**Graafi:** {len(entities)} entiteettiä, {len(relations)} relaatiota\n\n")
    f.write("---\n\n")

    f.write("## 📊 Yhteenveto\n\n")
    f.write("| Metriikka | Arvo |\n|-----------|------|\n")
    f.write(f"| **Kysymyksiä yhteensä** | {total_questions} kpl |\n")
    f.write(f"| **Graph-kysymykset** (mix/local/global) | {graph_count} kpl |\n")
    f.write(f"| **Chunk-kysymykset** (baseline/naive) | {chunk_count} kpl |\n")
    f.write(f"| **Entiteettiä graafissa** | {len(entities):,} |\n")
    f.write(f"| **Relaatiota graafissa** | {len(relations):,} |\n")
    f.write(f"| **Generointi kesti** | {generation_time:.1f}s |\n")
    f.write(f"| **Tokeneita yhteensä** | {all_tokens['total']:,} |\n")
    f.write(f"| **Kokonaiskustannus** | ${total_cost:.4f} |\n")
    f.write(f"| **Malli** | {MODEL} |\n\n")
    f.write("---\n\n")

    # Graph-kysymykset
    if graph_structured_data:
        f.write("## 🕸️ Graph-kysymykset (testaavat mix/local/global)\n\n")
        f.write("> Nämä kysymykset vaativat entiteettien välisten suhteiden ymmärtämistä "
                "– naive-haku ei pärjää hyvin näissä.\n\n")

        for user in graph_structured_data.get("users", []):
            f.write(f"### Käyttäjä {user['user_id']}: {user.get('description', '')}\n\n")
            for task in user.get("tasks", []):
                f.write(f"#### Tehtävä {task['task_id']}: {task.get('description', '')}\n\n")
                for q_idx, q in enumerate(task.get("questions", []), 1):
                    f.write(f"{q_idx}. {q}\n")
                f.write("\n")

        f.write("---\n\n")

    # Chunk-kysymykset
    if chunk_structured_data:
        f.write("## 📄 Chunk-kysymykset (baseline – toimii naivelle)\n\n")
        f.write("> Nämä kysymykset voidaan vastata tekstitiedoista – käytetään baseline-vertailuun.\n\n")

        for user in chunk_structured_data.get("users", []):
            f.write(f"### Käyttäjä {user['user_id']}: {user.get('description', '')}\n\n")
            for task in user.get("tasks", []):
                f.write(f"#### Tehtävä {task['task_id']}: {task.get('description', '')}\n\n")
                for q_idx, q in enumerate(task.get("questions", []), 1):
                    f.write(f"{q_idx}. {q}\n")
                f.write("\n")

        f.write("---\n\n")

    # Kaikki kysymykset taulukossa
    f.write("## 🗂️ Kaikki kysymykset taulukossa\n\n")
    f.write("| # | Tyyppi | Käyttäjä | Tehtävä | Kysymys |\n")
    f.write("|---|--------|----------|---------|----------|\n")
    for i, tc in enumerate(all_test_cases, 1):
        m   = tc["metadata"]
        q   = tc["question"].replace("|", "\\|")
        if len(q) > 100:
            q = q[:97] + "..."
        qtype = "🕸️ graph" if m["question_type"] == "graph" else "📄 chunk"
        f.write(f"| {i} | {qtype} | U{m['user_id']} | T{m['task_id']} | {q} |\n")

print(f"✓ Markdown tallennettu: {md_path}\n")

# ============================================================================
# LOPPUYHTEENVETO
# ============================================================================

print(f"{'='*80}")
print(f"VALMIS")
print(f"{'='*80}")
print(f"Graph-kysymykset:  {graph_count} kpl  ← testaavat mix/local/global")
print(f"Chunk-kysymykset:  {chunk_count} kpl  ← baseline / naive")
print(f"Yhteensä:          {total_questions} kpl")
print(f"JSON:              {output_path}")
print(f"Markdown:          {md_path}")
print(f"Kustannus:         ${total_cost:.4f}")
print(f"{'='*80}\n")
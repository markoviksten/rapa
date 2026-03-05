import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import time
from datetime import datetime, timedelta

load_dotenv()

# Luetaan API-avain .env-tiedostosta
api_key = os.getenv("LLM_BINDING_API_KEY")
if not api_key:
    raise ValueError("LLM_BINDING_API_KEY ei löydy .env-tiedostosta!")

client = OpenAI(api_key=api_key)

# Vektorit josta testcaset generoidaan
INPUT_FILE = "data/nano7/rag_storage/vdb_chunks.json"

# Token-hinnat (USD per 1M tokenia)
TOKEN_PRICES = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
}

# ============================================================================
# KONFIGURAATIO
# ============================================================================

# Malli
MODEL = "gpt-4o-mini"

# Kuinka monta chunkkia käytetään dataset-kuvaukseen (0 = kaikki)
DESCRIPTION_CHUNK_LIMIT = 30

# Käyttäjien, tehtävien ja kysymysten määrä
NUM_USERS     = 5   # Edge et al.: 5 käyttäjää
NUM_TASKS     = 5   # Edge et al.: 5 tehtävää per käyttäjä
NUM_QUESTIONS = 5   # Edge et al.: 5 kysymystä per (käyttäjä, tehtävä)

# Kieli jolla kysymykset generoidaan
LANGUAGE = "Finnish"

# ============================================================================
# Älä muokkaa tästä alaspäin
# ============================================================================

# Aseta hakemistot
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
print("EDGE ET AL. (2024) -MENETELMÄ – TESTIKYSYMYSTEN GENEROINTI")
print("=" * 80)

# ============================================================================
# VAIHE 1: Lue chunkit ja muodosta dataset-kuvaus
# ============================================================================

json_path = project_root / INPUT_FILE

try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["data"]
    print(f"\n✓ Ladattu {len(chunks)} chunkkia")
except Exception as e:
    print(f"\n❌ Virhe chunkkien lataamisessa: {e}")
    exit()

# Rakenna dataset-kuvaus chunkkeista (rajoitettu määrä)
sample_chunks = chunks if DESCRIPTION_CHUNK_LIMIT == 0 else chunks[:DESCRIPTION_CHUNK_LIMIT]
combined_text = "\n\n---\n\n".join(c["content"] for c in sample_chunks)

print(f"\nVaihe 1/3: Muodostetaan dataset-kuvaus ({len(sample_chunks)} chunkkia)...")

description_prompt = f"""Analyze the following document excerpts and write a concise description (3-5 sentences) 
of the dataset: what domain it covers, what kind of information it contains, and what it would typically be used for.

Document excerpts:
{combined_text[:12000]}

Write the description in English."""

try:
    desc_response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": description_prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    dataset_description = desc_response.choices[0].message.content.strip()
    desc_tokens = {
        "input":  desc_response.usage.prompt_tokens,
        "output": desc_response.usage.completion_tokens,
        "total":  desc_response.usage.total_tokens,
    }
    print(f"✓ Dataset-kuvaus luotu ({desc_tokens['total']} tokenia)")
    print(f"\n  Kuvaus: {dataset_description[:200]}...\n")
except Exception as e:
    print(f"❌ Virhe dataset-kuvauksessa: {e}")
    exit()

# ============================================================================
# VAIHE 2: Generoi käyttäjät, tehtävät ja kysymykset (master prompt)
# ============================================================================

print("Vaihe 2/3: Generoidaan käyttäjät, tehtävät ja kysymykset...")

master_prompt = f"""Given the following description of a dataset:

{dataset_description}

Please identify {NUM_USERS} potential users who would engage with this dataset. 
For each user, list {NUM_TASKS} tasks they would perform with this dataset. 
Then, for each (user, task) combination, generate {NUM_QUESTIONS} questions that require 
a high-level understanding of the entire dataset.

Important requirements:
- Questions must be in {LANGUAGE}
- Questions should require synthesizing information across the entire dataset, not just one section
- Questions should be open-ended and meaningful (no simple yes/no answers)
- Each question should reflect the specific user's perspective and task context

Return ONLY a JSON object in this exact structure:
{{
  "users": [
    {{
      "user_id": 1,
      "description": "User description",
      "tasks": [
        {{
          "task_id": 1,
          "description": "Task description",
          "questions": [
            "Question 1 in {LANGUAGE}",
            "Question 2 in {LANGUAGE}",
            "Question 3 in {LANGUAGE}",
            "Question 4 in {LANGUAGE}",
            "Question 5 in {LANGUAGE}"
          ]
        }}
      ]
    }}
  ]
}}"""

try:
    start_time = time.time()

    master_response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": master_prompt}],
        max_tokens=6000,
        temperature=0.8,
        response_format={"type": "json_object"},
    )

    master_tokens = {
        "input":  master_response.usage.prompt_tokens,
        "output": master_response.usage.completion_tokens,
        "total":  master_response.usage.total_tokens,
    }

    structured_data = json.loads(master_response.choices[0].message.content)
    end_time = time.time()
    duration = end_time - start_time

    print(f"✓ Rakenne generoitu ({master_tokens['total']} tokenia, {duration:.1f}s)\n")

except Exception as e:
    print(f"❌ Virhe master promptissa: {e}")
    exit()

# ============================================================================
# VAIHE 3: Muunna flat testcase-listaksi
# ============================================================================

print("Vaihe 3/3: Muunnetaan testcase-muotoon...")

all_test_cases = []
users = structured_data.get("users", [])

for user in users:
    user_id   = user.get("user_id")
    user_desc = user.get("description", "")

    for task in user.get("tasks", []):
        task_id   = task.get("task_id")
        task_desc = task.get("description", "")

        for q_idx, question in enumerate(task.get("questions", []), 1):
            all_test_cases.append({
                "question": question,
                "metadata": {
                    "user_id":          user_id,
                    "user_description": user_desc,
                    "task_id":          task_id,
                    "task_description": task_desc,
                    "question_index":   q_idx,
                    "method":           "edge_et_al_2024",
                }
            })

total_questions = len(all_test_cases)
print(f"✓ {total_questions} kysymystä muunnettu\n")

# ============================================================================
# KUSTANNUKSET
# ============================================================================

total_tokens = {
    "input":  desc_tokens["input"]  + master_tokens["input"],
    "output": desc_tokens["output"] + master_tokens["output"],
    "total":  desc_tokens["total"]  + master_tokens["total"],
}

prices      = TOKEN_PRICES.get(MODEL, {"input": 0, "output": 0})
input_cost  = (total_tokens["input"]  / 1_000_000) * prices["input"]
output_cost = (total_tokens["output"] / 1_000_000) * prices["output"]
total_cost  = input_cost + output_cost

# ============================================================================
# TALLENNUS – JSON
# ============================================================================

final_output = {
    "test_cases": all_test_cases,
    "metadata": {
        "method":                "edge_et_al_2024_graphrag",
        "dataset_description":   dataset_description,
        "total_questions":       total_questions,
        "num_users":             NUM_USERS,
        "num_tasks_per_user":    NUM_TASKS,
        "num_questions_per_task": NUM_QUESTIONS,
        "generation_time_seconds": duration,
        "generation_time_formatted": str(timedelta(seconds=int(duration))),
        "tokens_used":  total_tokens,
        "cost_estimate": {
            "input_cost":  input_cost,
            "output_cost": output_cost,
            "total_cost":  total_cost,
            "model":       MODEL,
        },
        "structure": structured_data,
    }
}

output_path = OUTPUT_FILE.absolute()

try:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"{'='*80}")
    print(f"TALLENNUS")
    print(f"{'='*80}")
    print(f"✓ JSON tallennettu: {output_path}")
    print(f"  Kysymyksiä: {total_questions}")
    print(f"  Tokeneita:  {total_tokens['total']:,}")
    print(f"  Kustannus:  ${total_cost:.4f}\n")

except Exception as e:
    print(f"❌ Virhe JSON-tallennuksessa: {e}")

# ============================================================================
# TALLENNUS – MARKDOWN-RAPORTTI
# ============================================================================

md_path = MD_FILE.absolute()

try:
    with open(MD_FILE, "w", encoding="utf-8") as f:

        f.write("# Testikysymysten Generointiraportti – Edge et al. (2024)\n\n")
        f.write(f"**Luotu:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Menetelmä:** Edge et al. (2024) – GraphRAG / LLM-as-a-Judge parivertailu\n\n")
        f.write(f"**Input:** `{INPUT_FILE}`\n\n")
        f.write(f"**Output:** `{OUTPUT_FILE.name}`\n\n")
        f.write("---\n\n")

        # Dataset-kuvaus
        f.write("## 📄 Dataset-kuvaus\n\n")
        f.write(f"{dataset_description}\n\n")
        f.write("---\n\n")

        # Yhteenveto
        f.write("## 📊 Yhteenveto\n\n")
        f.write("| Metriikka | Arvo |\n")
        f.write("|-----------|------|\n")
        f.write(f"| **Kysymyksiä yhteensä** | {total_questions} kpl |\n")
        f.write(f"| **Käyttäjiä** | {NUM_USERS} |\n")
        f.write(f"| **Tehtäviä / käyttäjä** | {NUM_TASKS} |\n")
        f.write(f"| **Kysymyksiä / tehtävä** | {NUM_QUESTIONS} |\n")
        f.write(f"| **Generointi kesti** | {duration:.1f}s |\n")
        f.write(f"| **Tokeneita yhteensä** | {total_tokens['total']:,} |\n")
        f.write(f"| **Kokonaiskustannus** | ${total_cost:.4f} |\n")
        f.write(f"| **Malli** | {MODEL} |\n\n")
        f.write("---\n\n")

        # Kysymykset rakenteellisesti
        f.write("## 📝 Generoidut Kysymykset\n\n")

        for user in users:
            user_id   = user.get("user_id")
            user_desc = user.get("description", "")
            f.write(f"### Käyttäjä {user_id}: {user_desc}\n\n")

            for task in user.get("tasks", []):
                task_id   = task.get("task_id")
                task_desc = task.get("description", "")
                f.write(f"#### Tehtävä {task_id}: {task_desc}\n\n")

                for q_idx, question in enumerate(task.get("questions", []), 1):
                    f.write(f"{q_idx}. {question}\n")
                f.write("\n")

        f.write("---\n\n")

        # Flat taulukko
        f.write("## 🗂️ Kaikki kysymykset taulukossa\n\n")
        f.write("| # | Käyttäjä | Tehtävä | Kysymys |\n")
        f.write("|---|----------|---------|----------|\n")

        for i, tc in enumerate(all_test_cases, 1):
            m        = tc["metadata"]
            question = tc["question"].replace("|", "\\|")
            if len(question) > 100:
                question = question[:97] + "..."
            f.write(f"| {i} | U{m['user_id']} | T{m['task_id']} | {question} |\n")

    print(f"✓ Markdown-raportti tallennettu: {md_path}\n")

except Exception as e:
    print(f"❌ Virhe Markdown-tallennuksessa: {e}\n")

# ============================================================================
# LOPPUYHTEENVETO
# ============================================================================

print(f"{'='*80}")
print(f"VALMIS")
print(f"{'='*80}")
print(f"Käyttäjiä:         {len(users)}")
print(f"Kysymyksiä:        {total_questions}")
print(f"JSON:              {output_path}")
print(f"Markdown-raportti: {md_path}")
print(f"Kustannus:         ${total_cost:.4f}")
print(f"{'='*80}\n")
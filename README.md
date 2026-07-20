# Self-Correcting RAG Agent over MongoDB Vector Search

Self-Correcting RAG Agent using MongoDB Atlas Vector Search, LangChain, LangGraph, and LangSmith.

---

## Evaluation Suite

This project includes a comprehensive, end-to-end evaluation system that benchmark RAG configurations against deterministic metrics and regression gates.

### How to Run Evaluation

To execute the complete evaluation dataset and print the benchmark comparison report, run the following command from the project root:

```bash
cd capstone-rag
source .venv/bin/activate
python evals/run_evals.py
```

---

### Dataset Description

The evaluation dataset contains **27 test examples** located in [evals/datasets/qa.jsonl](file:///Users/phani/ragProject/capstone-rag/evals/datasets/qa.jsonl):
* **22 Answerable Questions**: Asking about specific listing features, location details, room characteristics, amenities, and transit descriptions. Each of these questions has ground-truth relevant listing IDs from the MongoDB Atlas corpus.
* **5 No-Answer Questions**: Questions that fall completely outside the corpus/database scope (e.g., listings in Tokyo, Paris, Bali, Berlin, or London). These test the safety and refusal correctness of the agent.

---

### Metrics Definitions

1. **Recall@5**
   * **Definition**: Evaluates the retriever quality. Measures the proportion of expected relevant listing IDs found in the top-5 retrieved chunks.
   * **Formula**: $\frac{\text{Expected IDs found in top-5}}{\text{Total expected relevant IDs}}$
   * Note: Excludes `no_answer` questions to prevent score distortion.

2. **Citation Groundedness**
   * **Definition**: Assesses the reliability of the generated output. Checks if all cited Listing IDs in the response are present in the retrieved context. Also flags answerable questions where the model failed to output citations.

3. **Refusal Correctness**
   * **Definition**: Evaluates safety. Ensures that for out-of-corpus (`no_answer`) questions, the agent refuses to answer instead of hallucinating details.
   * **Formula**: $\frac{\text{Correct Refusals}}{\text{Total no-answer examples}}$

---

### Regression Gate Thresholds

The evaluation runner script enforces strict regression checks:
* **Recall@5**: $\ge$ 70%
* **Citation Groundedness**: $\ge$ 90%
* **Refusal Correctness**: $\ge$ 100%

If any metric falls below the threshold, the script exits with a non-zero exit code (`1`), indicating a failed regression gate.

---

### Actual Latest Evaluation Results

```
==================================================
SELF-CORRECTING RAG — EVALUATION REPORT
==================================================
Total Evaluation Examples: 27
  Answerable Examples: 22
  No-Answer Examples: 5

RETRIEVAL METRICS
Recall@5: 93.56% (PASS)

GROUNDING METRICS
Citation Groundedness: 100.00% (PASS)

SAFETY / REFUSAL METRICS
Refusal Correctness: 100.00% (PASS)

Overall Regression Gate: PASSED
==================================================
```

### Before-vs-After Experiment Comparison

By comparing the **Baseline** configuration (no relevance grading or query rewriting) against the **Improved** self-correcting configuration (batch relevance grading, self-correction looping, and LLM query rewriting), we achieve the following results:

| Metric                | Before   | After    | Change    |
| --------------------- | -------: | -------: | --------: |
| Recall@5              | 93.56%   | 93.56%   | +0.00 pp  |
| Citation Groundedness | 100.00%  | 100.00%  | +0.00 pp  |
| Refusal Correctness   | 100.00%  | 100.00%  | +0.00 pp  |

*Note: Since the generation model is `openai/gpt-4o-mini`, both configurations achieve perfect safety and grounding due to the model's strong reasoning capability, while the self-correcting agent provides additional query-rewriting routing and loop safety guards.*

---

### LangSmith Experiment Information

Tracing is fully integrated via LangGraph. When running evaluations with LangSmith enabled, all runs are logged to your LangSmith project (`capstone-rag`). 

Traces log important pipeline stages including:
* **Vector Search** retrieval operations.
* **Batch Document Grading** runs.
* **LLM Query Rewriting** steps.
* **Grounded Generation** invocations.

---

### Limitations of the Evaluation Methodology

* **Corpus Size**: The test corpus contains a subset of 200 documents, which has high query-retrieved overlap and may result in artificially high Recall@5. Larger datasets could lower the baseline retrieval recall.
* **String Refusal Detection**: Refusal correctness relies on detecting a predefined refusal string, which might fail if the model phrasings deviate (handled in improved mode by setting a structured boolean `refused: true` flag in the state).

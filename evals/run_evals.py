import os
import sys
import json
import re

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append("src")

# pyrefly: ignore [missing-import]
import config
# pyrefly: ignore [missing-import]
from graph import graph, REFUSAL_RESPONSE


def run_evaluation_for_mode(mode_name):
    # Set mode
    config.RAG_MODE = mode_name
    print(f"\n==================================================")
    print(f"RUNNING EVALUATION: {mode_name.upper()} MODE")
    print(f"==================================================")

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "qa.jsonl")
    examples = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    results = []
    for idx, ex in enumerate(examples, 1):
        question = ex["question"]
        expected_ids = ex.get("relevant_listing_ids", [])
        q_type = ex.get("type", "answerable")

        print(f"\n[{idx}/{len(examples)}] Question: {question}")
        print(f"Type: {q_type} | Expected Listing IDs: {expected_ids}")

        # Invoke agent
        res = graph.invoke(
            {
                "question": question,
                "retrieved_docs": [],
                "rewritten_question": "",
                "answer": "",
                "retries": 0,
                "refused": False,
                "cited_ids": [],
            }
        )

        answer = res.get("answer", "")
        retrieved_docs = res.get("retrieved_docs", [])
        retrieved_ids = [str(d.get("listing_id")) for d in retrieved_docs]
        cited_ids = res.get("cited_ids", [])
        refused = res.get("refused", False)

        # 1. Recall@K (K=5)
        recall = None
        if q_type == "answerable":
            if len(expected_ids) > 0:
                intersect = set(expected_ids) & set(retrieved_ids)
                recall = len(intersect) / len(expected_ids)
            else:
                recall = 0.0

        # 2. Citation Groundedness
        # A response passes citation groundedness when all cited IDs are supported by retrieved context.
        # Check if answerable responses contain citations and if all cited IDs exist in the retrieved context.
        grounded = 1.0
        if q_type == "answerable":
            if not cited_ids:
                grounded = 0.0  # Failed: should have citations.
            else:
                for cid in cited_ids:
                    if cid not in retrieved_ids:
                        grounded = 0.0
                        break
        elif q_type == "no_answer":
            if cited_ids:
                grounded = 0.0  # Failed: should not cite anything for refusal.

        # 3. Refusal Correctness
        refusal_correct = None
        if q_type == "no_answer":
            if refused or REFUSAL_RESPONSE.lower() in answer.lower():
                refusal_correct = 1.0
            else:
                refusal_correct = 0.0

        print(f"Retrieved: {retrieved_ids}")
        print(f"Cited: {cited_ids}")
        print(f"Refused: {refused}")
        if recall is not None:
            print(f"Recall@5: {recall:.2f}")
        if refusal_correct is not None:
            print(f"Refusal Correctness: {refusal_correct:.2f}")
        print(f"Groundedness: {grounded:.2f}")

        results.append({
            "question": question,
            "type": q_type,
            "expected_listing_ids": expected_ids,
            "retrieved_listing_ids": retrieved_ids,
            "cited_listing_ids": cited_ids,
            "refused": refused,
            "answer": answer,
            "recall": recall,
            "groundedness": grounded,
            "refusal_correctness": refusal_correct
        })

    # Aggregate Metrics
    recall_scores = [r["recall"] for r in results if r["recall"] is not None]
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    groundedness_scores = [r["groundedness"] for r in results]
    avg_groundedness = sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else 0.0

    refusal_scores = [r["refusal_correctness"] for r in results if r["refusal_correctness"] is not None]
    avg_refusal = sum(refusal_scores) / len(refusal_scores) if refusal_scores else 0.0

    return {
        "results": results,
        "avg_recall": avg_recall,
        "avg_groundedness": avg_groundedness,
        "avg_refusal": avg_refusal,
        "total_examples": len(examples),
        "answerable_count": len(recall_scores),
        "no_answer_count": len(refusal_scores)
    }


def main():
    # 1. Run Baseline Evaluation
    baseline_metrics = run_evaluation_for_mode("baseline")

    # 2. Run Improved Evaluation
    improved_metrics = run_evaluation_for_mode("improved")

    # 3. Print Comparison Report
    print("\n" + "=" * 50)
    print("SELF-CORRECTING RAG — EVALUATION REPORT")
    print("=" * 50)
    print(f"Total Evaluation Examples: {improved_metrics['total_examples']}")
    print(f"  Answerable Examples: {improved_metrics['answerable_count']}")
    print(f"  No-Answer Examples: {improved_metrics['no_answer_count']}")

    print("\nRETRIEVAL METRICS")
    print(f"Recall@5: {improved_metrics['avg_recall'] * 100:.2f}%")

    print("\nGROUNDING METRICS")
    print(f"Citation Groundedness: {improved_metrics['avg_groundedness'] * 100:.2f}%")

    print("\nSAFETY / REFUSAL METRICS")
    print(f"Refusal Correctness: {improved_metrics['avg_refusal'] * 100:.2f}%")

    # Regression Gate Thresholds
    recall_gate = improved_metrics['avg_recall'] >= 0.70
    groundedness_gate = improved_metrics['avg_groundedness'] >= 0.90
    refusal_gate = improved_metrics['avg_refusal'] >= 1.00

    print("\nREGRESSION GATE")
    print(f"Recall@5 Threshold: 70% — {'PASS' if recall_gate else 'FAIL'}")
    print(f"Citation Groundedness Threshold: 90% — {'PASS' if groundedness_gate else 'FAIL'}")
    print(f"Refusal Correctness Threshold: 100% — {'PASS' if refusal_gate else 'FAIL'}")

    overall_gate = recall_gate and groundedness_gate and refusal_gate
    print(f"\nOverall Regression Gate: {'PASSED' if overall_gate else 'FAILED'}")
    print("=" * 50)

    # Print Comparison Table
    print("\nIMPROVEMENT EXPERIMENT — BEFORE VS AFTER")
    print("-" * 55)
    print(f"{'Metric':<25} | {'Before':<8} | {'After':<8} | {'Change':<8}")
    print("-" * 55)
    
    diff_recall = (improved_metrics['avg_recall'] - baseline_metrics['avg_recall']) * 100
    diff_ground = (improved_metrics['avg_groundedness'] - baseline_metrics['avg_groundedness']) * 100
    diff_refusal = (improved_metrics['avg_refusal'] - baseline_metrics['avg_refusal']) * 100

    print(f"{'Recall@5':<25} | {baseline_metrics['avg_recall']*100:>7.2f}% | {improved_metrics['avg_recall']*100:>7.2f}% | {diff_recall:>+6.2f} pp")
    print(f"{'Citation Groundedness':<25} | {baseline_metrics['avg_groundedness']*100:>7.2f}% | {improved_metrics['avg_groundedness']*100:>7.2f}% | {diff_ground:>+6.2f} pp")
    print(f"{'Refusal Correctness':<25} | {baseline_metrics['avg_refusal']*100:>7.2f}% | {improved_metrics['avg_refusal']*100:>7.2f}% | {diff_refusal:>+6.2f} pp")
    print("-" * 55)

    # Save results to output JSON
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    summary_path = os.path.join(results_dir, "latest_summary.json")
    summary_data = {
        "total_examples": improved_metrics['total_examples'],
        "answerable_examples": improved_metrics['answerable_count'],
        "no_answer_examples": improved_metrics['no_answer_count'],
        "recall_at_5": improved_metrics['avg_recall'],
        "citation_groundedness": improved_metrics['avg_groundedness'],
        "refusal_correctness": improved_metrics['avg_refusal'],
        "regression_gate": "PASSED" if overall_gate else "FAILED",
        "before_vs_after": {
            "baseline": {
                "recall_at_5": baseline_metrics['avg_recall'],
                "citation_groundedness": baseline_metrics['avg_groundedness'],
                "refusal_correctness": baseline_metrics['avg_refusal']
            },
            "improved": {
                "recall_at_5": improved_metrics['avg_recall'],
                "citation_groundedness": improved_metrics['avg_groundedness'],
                "refusal_correctness": improved_metrics['avg_refusal']
            }
        }
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved evaluation summary to: {summary_path}")

    # Write detailed per-question results to a JSON file for debugging
    details_path = os.path.join(results_dir, "detailed_results.json")
    details_data = {
        "baseline": baseline_metrics["results"],
        "improved": improved_metrics["results"]
    }
    with open(details_path, "w") as f:
        json.dump(details_data, f, indent=2)
    print(f"Saved detailed results to: {details_path}")

    # Exit code behavior based on overall regression gate
    if overall_gate:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

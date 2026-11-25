import os
import sys
import json
import re
import string
from pathlib import Path

def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and stripping punctuation and whitespace."""
    text = str(text).lower().strip()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def is_closed_question(question: str, ground_truth: str) -> bool:
    """Determine whether a question is closed-form (Yes/No / choices) or open-form."""
    gt_norm = normalize_text(ground_truth)
    if gt_norm in ["yes", "no"]:
        return True
    
    q_lower = question.lower().strip()
    closed_starters = [
        "is ", "are ", "was ", "were ", "do ", "does ", "did ", 
        "can ", "could ", "has ", "have ", "had ", "will ", "would "
    ]
    if any(q_lower.startswith(s) for s in closed_starters):
        return True
    if " or " in q_lower and ("?" in q_lower or len(q_lower.split()) < 15):
        return True
    return False

def compute_closed_accuracy(ground_truth: str, pred_answer: str) -> float:
    """
    Accuracy for closed-form questions.
    Checks exact match or presence of closed answer in normalized prediction.
    """
    gt_norm = normalize_text(ground_truth)
    pred_norm = normalize_text(pred_answer)
    
    if gt_norm in ["yes", "no"]:
        pred_words = pred_norm.split()
        if gt_norm in pred_words:
            # check if contradicting word is not also present or prioritized
            if gt_norm == "yes" and "no" not in pred_words:
                return 1.0
            elif gt_norm == "no" and "yes" not in pred_words:
                return 1.0
            elif pred_words and pred_words[0] == gt_norm:
                return 1.0
            return 0.0
        return 1.0 if pred_norm == gt_norm else 0.0

    # General exact match
    if pred_norm == gt_norm or gt_norm in pred_norm:
        return 1.0
    return 0.0

def compute_open_recall(ground_truth: str, pred_answer: str) -> float:
    """
    Token-level recall for open-form questions:
    Measures the proportion of ground-truth tokens present in the generated sequence.
    """
    gt_tokens = set(normalize_text(ground_truth).split())
    if not gt_tokens:
        return 1.0
    pred_tokens = set(normalize_text(pred_answer).split())
    matched = gt_tokens.intersection(pred_tokens)
    return len(matched) / len(gt_tokens)

def evaluate_sample(question: str, ground_truth: str, pred_answer: str, q_type: str = None):
    if q_type is None:
        q_type = "CLOSED" if is_closed_question(question, ground_truth) else "OPEN"
    else:
        q_type = q_type.upper()

    if q_type == "CLOSED":
        score = compute_closed_accuracy(ground_truth, pred_answer)
        metric_name = "accuracy"
    else:
        score = compute_open_recall(ground_truth, pred_answer)
        metric_name = "recall"

    return {
        "question": question,
        "ground_truth": ground_truth,
        "pred_answer": pred_answer,
        "type": q_type,
        "metric": metric_name,
        "score": score
    }

def summarize_evaluation(results: list) -> dict:
    closed_scores = [r["score"] for r in results if r["type"] == "CLOSED"]
    open_scores = [r["score"] for r in results if r["type"] == "OPEN"]
    
    closed_acc = (sum(closed_scores) / len(closed_scores) * 100.0) if closed_scores else 0.0
    open_rec = (sum(open_scores) / len(open_scores) * 100.0) if open_scores else 0.0
    overall = (sum(r["score"] for r in results) / len(results) * 100.0) if results else 0.0

    return {
        "total_samples": len(results),
        "closed_samples": len(closed_scores),
        "open_samples": len(open_scores),
        "closed_accuracy": round(closed_acc, 2),
        "open_recall": round(open_rec, 2),
        "overall_score": round(overall, 2),
    }

if __name__ == "__main__":
    print("VQA-RAD Evaluation module loaded successfully.")

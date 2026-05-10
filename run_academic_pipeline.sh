#!/bin/bash
source venv/bin/activate

echo "========================================================"
echo "🚀 INITIATING ACADEMIC-GRADE BTP PIPELINE 🚀"
echo "========================================================"
python -u fetch_academic_datasets.py
echo ">> PHASE 1: Training on ProntoQA Logic Manifold..."
python -u train_academic.py

echo ">> PHASE 2: Evaluating on LogiQA Benchmark..."
python -u evaluate_academic.py

echo "========================================================"
echo "✅ PIPELINE COMPLETE ✅"

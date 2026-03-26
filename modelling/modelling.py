from model.randomforest import RandomForest
from model.svm import SVM
from model.logistic_regression import LogisticReg

from sklearn.metrics import accuracy_score, f1_score

MODEL_REGISTRY = {
    'Random Forest'       : RandomForest,
    'SVM'                 : SVM,
    'Logistic Regression' : LogisticReg,
}

def run_single_model(ModelClass, name: str, data):
    # Instantiate, train, predict and evaluate one model. Returns accuracy & F1.
    model = ModelClass(
        model_name=name,
        embeddings=data.get_X_train(),
        y=data.get_type_y_train(),
    )
    model.train(data)
    model.predict(data.get_X_test())
    model.print_results(data)
 
    acc = accuracy_score(data.get_type_y_test(), model.predictions)
    f1  = f1_score(data.get_type_y_test(), model.predictions,
                   average='weighted', zero_division=0)
    return acc, f1

def model_predict(data, df, name: str):
    results = []
 
    for model_name, ModelClass in MODEL_REGISTRY.items():
        full_name = f"{model_name} | {name}"
        print(f"\n{'='*60}")
        print(f"  {full_name}")
        print(f"{'='*60}")
 
        acc, f1 = run_single_model(ModelClass, full_name, data)
        results.append((model_name, acc, f1))
 
    return results
 
 
def model_evaluate(model, data):
    # Print the classification report for a single trained model.
    model.print_results(data)
 
 
def print_comparison_table(all_results: dict):
    # Print a formatted comparison table.
    
    chain_levels = list(all_results.keys())
    model_names  = [r[0] for r in list(all_results.values())[0]]
 
    col_w   = 22
    num_w   = 12
    headers = ['Model'] + chain_levels
    sep     = '-' * (col_w + num_w * len(chain_levels) * 2 + 4)
 
    print("\n")
    print("=" * len(sep))
    print("  MODEL ACCURACY COMPARISON (weighted F1 / Accuracy)")
    print("=" * len(sep))
 
    # Header row
    header_line = f"{'Model':<{col_w}}"
    for lvl in chain_levels:
        short = lvl[:num_w*2-2]
        header_line += f"  {short:<{num_w*2-2}}"
    print(header_line)
    print("-" * len(sep))
 
    # One row per model
    for i, model_name in enumerate(model_names):
        row = f"{model_name:<{col_w}}"
        for lvl in chain_levels:
            acc = all_results[lvl][i][1]
            f1  = all_results[lvl][i][2]
            cell = f"F1={f1:.2f} Acc={acc:.2f}"
            row += f"  {cell:<{num_w*2-2}}"
        print(row)

    print("=" * len(sep))

    # Best model per chain level
    print("\n  >> Best model per level (by weighted F1):")
    for lvl in chain_levels:
        best = max(all_results[lvl], key=lambda x: x[2])
        print(f"     {lvl:<35} -> {best[0]}  (F1={best[2]:.2f}, Acc={best[1]:.2f})")
    print()
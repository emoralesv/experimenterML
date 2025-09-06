import random

def random_evaluation(models, size):
    results = []
    for model in models:
        for s in size:
            accuracy = round(random.uniform(0.5, 1.0), 4)
            results.append({
                "model": model,
                "size": s,
                "accuracy": accuracy
            })
    return results
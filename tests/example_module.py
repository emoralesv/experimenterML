import random
import time
def random_evaluation(**kwargs):
    accuracy = round(random.uniform(0.5, 1.0),1)

    time.sleep(1)  # Simulate a time-consuming evaluation
    return {**kwargs, "accuracy": accuracy}
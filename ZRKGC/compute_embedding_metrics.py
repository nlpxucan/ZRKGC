from nlgeval import compute_metrics
import sys
metrics_dict = compute_metrics(hypothesis=sys.argv[1],
                               references=[sys.argv[2]],no_skipthoughts=True)
print(metrics_dict)

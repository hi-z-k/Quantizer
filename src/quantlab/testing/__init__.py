"""Testing and evaluation package for quantlab.

Provides dataset loaders and evaluation metrics for assessing quantized models:
- WikiText-2 non-overlapping token window evaluation: WikiText2Dataset
- Language model perplexity measurement: PerplexityMetric
- Weight error analysis (MSE, SQNR, worst-layer detection): WeightErrorMetric
- Model size estimation and real packed tensor measurement: estimate_size, measure_size, SizeReport
"""

from quantlab.testing.wikitext2_dataset import WikiText2Dataset
from quantlab.testing.perplexity_metric import PerplexityMetric
from quantlab.testing.weight_error_metric import WeightErrorMetric
from quantlab.testing.model_size_metric import estimate_size, measure_size, SizeReport

__all__ = [
    "WikiText2Dataset",
    "PerplexityMetric",
    "WeightErrorMetric",
    "estimate_size",
    "measure_size",
    "SizeReport",
]

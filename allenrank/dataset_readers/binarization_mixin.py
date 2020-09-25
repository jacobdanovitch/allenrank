import numpy as np
import logging
logger = logging.getLogger(__name__)

class BinarizationMixin:
    GRANULARITY_OPTIONS = ['passthrough', 'continuous', 'binary', 'argmax', 'trec']

    def __init__(self, granularity: str = 'continuous', threshold: float = None):
        assert granularity in BinarizationMixin.GRANULARITY_OPTIONS, f'Invalid granularity: {granularity}.'
        if granularity == 'binary' and threshold is not None:
            logger.warning(f'Threshold={threshold} provided with granularity={granularity}. Ignoring threshold.')
        
        self._granularity = granularity
        self._threshold = threshold

    def _process_label(self, label):
        if self._threshold is None or self._granularity in ['passthrough', 'continuous', 'argmax']:
            return label
        
        return int(label >= self._threshold)

    def _process_labels(self, labels):
        labels = list(map(self._process_label, labels))
        if self._granularity == 'argmax':
            _labels = np.zeros((len(labels),))
            _labels[np.argmax(labels)] = 1
            return _labels
            
        return labels
import logging
logger = logging.getLogger(__name__)

class BinarizationMixin:
    GRANULARITY_OPTIONS = ['passthrough', 'continuous', 'binary', 'trec']

    def __init__(self, granularity: str = 'continuous', threshold: float = None):
        assert granularity in BinarizationMixin.GRANULARITY_OPTIONS, f'Invalid granularity: {granularity}.'
        if granularity == 'binary' and threshold is not None: # not in ['binary', 'passthrough'] 
            logger.warning(f'Threshold={threshold} provided with granularity={granularity}. Ignoring threshold.')
        
        self._granularity = granularity
        self._threshold = threshold

    def _process_label(self, label):
        if self._threshold is None or self._granularity in ['passthrough', 'continuous']:
            return label
        
        return int(label >= self._threshold)
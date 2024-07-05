class BatchLoader:
    def __init__(self, data_x: list, data_y: list, batch_size: int = 1):
        self._data_x = data_x
        self._data_y = data_y
        self._batch_size = batch_size

    def __iter__(self):
        total_batches = len(self._data_x) // self._batch_size
        if len(self._data_x) % self._batch_size != 0:
            total_batches += 1

        for i in range(total_batches):
            start_idx = i * self._batch_size
            end_idx = start_idx + self._batch_size
            batch_x = self._data_x[start_idx:end_idx]
            batch_y = self._data_y[start_idx:end_idx]
            yield (batch_x, batch_y)

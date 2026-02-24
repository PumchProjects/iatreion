from time import perf_counter_ns


class Timer:
    def __enter__(self):
        self.start = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = perf_counter_ns()
        self.duration = (self.end - self.start) / 1e9

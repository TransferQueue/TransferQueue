import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.INFO))

TQ_PERF_LOG_FLUSH_INTERVAL = float(os.environ.get("TQ_PERF_LOG_FLUSH_INTERVAL", 10))  # in seconds


class IntervalPerfMonitor:
    def __init__(self, caller_name: str):
        self.caller_name = caller_name
        self.last_flush_time = time.perf_counter()

        self.success_counts: dict[str, int] = defaultdict(int)
        self.process_time: dict[str, list[float]] = defaultdict(list)

    def _flush_logs(self):
        now = time.perf_counter()

        # only flush if the interval has passed
        if (now - self.last_flush_time) >= TQ_PERF_LOG_FLUSH_INTERVAL:
            minutes = (now - self.last_flush_time) / 60

            total_requests = sum(self.success_counts.values())
            total_process_time = sum(sum(time_list) for time_list in self.process_time.values())
            total_avg_process_time = total_process_time / total_requests if total_requests > 0 else 0.0

            # max/min/avg time for each operation type
            op_detail_stats = []
            for op_type, count in self.success_counts.items():
                times = self.process_time[op_type]
                if not times:
                    op_avg = op_max = op_min = 0.0
                else:
                    op_avg = sum(times) / len(times)
                    op_max = max(times)
                    op_min = min(times)

                op_detail_stats.append(
                    f"{op_type}: req_count={count}, req/min={count / minutes:.2f}, "
                    f"avg_time={op_avg:.6f}s, max_time={op_max:.6f}s, min_time={op_min:.6f}s"
                )

            log_msg = (
                f"{self.caller_name}: [Performance] "
                f"Total success requests: {total_requests}, "
                f"Total req/min: {total_requests / minutes:.2f}, "
                f"Total avg process time: {total_avg_process_time:.4f}s; \n"
                f"Time range: last {minutes:.2f} minutes; \n"
                f"Per-operation statistics: {'; '.join(op_detail_stats)}"
            )

            logger.info(log_msg)

            # reset counts
            self.success_counts.clear()
            self.process_time.clear()
            self.last_flush_time = now

    @contextmanager
    def measure(self, op_type: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            cost = time.perf_counter() - start_time
            self.success_counts[op_type] += 1
            self.process_time[op_type].append(cost)

            # try flush logs
            self._flush_logs()

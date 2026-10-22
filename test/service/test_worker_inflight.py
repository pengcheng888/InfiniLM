import unittest

from infinilm.cluster.routing import WorkerInflightTracker


class WorkerInflightTrackerTest(unittest.TestCase):
    def test_begin_and_finish_track_active_requests(self):
        """请求生命周期驱动每个 worker 的活跃计数。"""
        tracker = WorkerInflightTracker()

        self.assertEqual(tracker.count("worker-1"), 0)
        self.assertFalse(tracker.has_inflight_requests("worker-1"))

        self.assertEqual(tracker.begin_request("worker-1"), 1)
        self.assertEqual(tracker.begin_request("worker-1"), 2)
        self.assertTrue(tracker.has_inflight_requests("worker-1"))

        self.assertEqual(tracker.finish_request("worker-1"), 1)
        self.assertEqual(tracker.finish_request("worker-1"), 0)
        self.assertFalse(tracker.has_inflight_requests("worker-1"))

    def test_repeated_finish_does_not_go_negative(self):
        """重复 finish 只会停留在 0，避免异常路径产生负 inflight。"""
        tracker = WorkerInflightTracker()

        tracker.begin_request("worker-1")
        tracker.finish_request("worker-1")

        self.assertEqual(tracker.finish_request("worker-1"), 0)
        self.assertEqual(tracker.finish_request("unknown"), 0)
        self.assertEqual(tracker.count("worker-1"), 0)

    def test_snapshot_omits_zero_counts(self):
        """归零后的 worker 不保留在活跃快照中。"""
        tracker = WorkerInflightTracker()

        tracker.begin_request("worker-1")
        tracker.begin_request("worker-2")
        tracker.finish_request("worker-1")

        self.assertEqual(tracker.snapshot(), {"worker-2": 1})


if __name__ == "__main__":
    unittest.main()

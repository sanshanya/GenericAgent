import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import ga


def parent():
    return SimpleNamespace(
        llmclient=SimpleNamespace(backend=SimpleNamespace(history=[])),
        get_ctx_multiplier=lambda: 1.0,
        task_dir="",
        extrakeyinfo=None,
        intervene=None,
        verbose=False,
        _turn_end_hooks={},
    )


def exhaust(generator):
    try:
        while True:
            next(generator)
    except StopIteration as stopped:
        return stopped.value


class MemoryPathTests(unittest.TestCase):
    def test_memory_guidance_is_bound_to_checkout(self):
        expected = Path(ga.memory_root).as_posix()
        previous = os.getcwd()
        other_cwd = tempfile.TemporaryDirectory()
        try:
            os.chdir(other_cwd.name)
            prompt = ga.get_global_memory()
        finally:
            os.chdir(previous)
            other_cwd.cleanup()

        self.assertIn(f"[Memory] ({expected})", prompt)
        self.assertIn(f"{expected}/global_mem.txt", prompt)
        self.assertIn(f"{expected}/memory_management_sop.md", prompt)
        self.assertNotIn("../memory", prompt)
        self.assertNotIn("./memory", prompt)
        self.assertNotIn("cwd =", prompt)

    def test_memory_sop_references_use_the_same_root(self):
        sop = Path(ga.memory_root, "memory_management_sop.md").read_text(encoding="utf-8")
        bound = ga.bind_memory_paths(sop)
        self.assertIn(Path(ga.memory_root).as_posix(), bound)
        self.assertNotIn("../memory", bound)
        self.assertNotIn("./memory", bound)

    def test_agent_may_settle_verified_memory_before_turn_ten(self):
        handler = ga.GenericAgentHandler(parent(), [], ".")
        handler.current_turn = 1
        task = handler.do_start_long_term_update({}, None)
        next(task)
        with self.assertRaises(StopIteration) as stopped:
            next(task)
        outcome = stopped.exception.value
        self.assertIn(Path(ga.memory_root).as_posix(), f"{outcome.data}\n{outcome.next_prompt}")
        handler.finish_task()

    def test_memory_settlement_and_inline_eval_are_process_safe(self):
        first = ga.GenericAgentHandler(parent(), [], ".")
        exhaust(first.do_start_long_term_update({}, None))
        settled = threading.Event()

        def settle():
            second = ga.GenericAgentHandler(parent(), [], ".")
            exhaust(second.do_start_long_term_update({}, None))
            second.finish_task()
            settled.set()

        thread = threading.Thread(target=settle)
        thread.start()
        self.assertFalse(settled.wait(0.05))
        first.finish_task()
        self.assertTrue(settled.wait(1))
        thread.join()

        active = {"count": 0, "max": 0}
        lock = threading.Lock()
        def probe():
            with lock:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
            time.sleep(0.05)
            with lock:
                active["count"] -= 1
            return "done"

        ga._test_inline_probe = probe
        handlers = [ga.GenericAgentHandler(parent(), [], ".") for _ in range(2)]
        def inline(handler):
            exhaust(handler.do_code_run(
                {"type": "python", "code": "__import__('ga')._test_inline_probe()", "inline_eval": True},
                None,
            ))
        threads = [threading.Thread(target=inline, args=(handler,)) for handler in handlers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        del ga._test_inline_probe
        self.assertEqual(active["max"], 1)

    def test_external_memory_root_is_bootstrapped_and_used_for_stats(self):
        with tempfile.TemporaryDirectory() as root:
            env = os.environ.copy()
            env["GA_MEMORY_ROOT"] = root
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import agentmain, ga; "
                    "assert ga.memory_root == __import__('os').path.abspath(__import__('os').environ['GA_MEMORY_ROOT']); "
                    "assert __import__('os').path.isfile(__import__('os').path.join(ga.memory_root, 'memory_management_sop.md')); "
                    "ga.log_memory_access(__import__('os').path.join(ga.memory_root, 'memory_management_sop.md')); "
                    "assert __import__('os').path.isfile(__import__('os').path.join(ga.memory_root, 'file_access_stats.json'))",
                ],
                env=env,
                check=True,
            )


if __name__ == "__main__":
    unittest.main()

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agentmain
import llmcore


class Handler:
    def __init__(self, parent, history, cwd):
        self.parent = parent
        self.history_info = history
        self.cwd = cwd
        self.working = {}
        self.code_stop_signal = []
        self.finished = False

    def finish_task(self):
        self.finished = True


def fake_agent():
    agent = object.__new__(agentmain.GenericAgent)
    agent.history = []
    agent.handler = None
    agent.extra_sys_prompts = [" native-extra"]
    agent.llmclient = SimpleNamespace(
        backend=SimpleNamespace(
            extra_sys_prompt=" backend-extra",
            config_name="native_oai_config",
        ),
        log_path="",
    )
    agent.log_path = "test.log"
    agent.peer_hint = False
    agent.force_non_stream = False
    agent.verbose = False
    agent.is_running = False
    agent.stop_sig = False
    return agent


class ExecuteTaskTests(unittest.TestCase):
    def test_model_call_reuses_config_without_agent_history(self):
        agent = fake_agent()
        agent.llmclient.backend.history = ["unchanged"]
        with patch.object(agentmain, "fast_ask", return_value="review") as call:
            self.assertEqual(agent.model_call("gate prompt"), "review")
        call.assert_called_once_with("gate prompt", "native_oai_config", temperature=0)
        self.assertEqual(agent.llmclient.backend.history, ["unchanged"])

    def test_model_tool_call_returns_isolated_structured_calls(self):
        agent = fake_agent()
        agent.llmclient.backend.history = ["unchanged"]

        class Session:
            temperature = 1
            tools = None
            tool_choice = None

            def raw_ask(self, messages):
                self.messages = messages
                if False:
                    yield None
                return [
                    {"type": "thinking", "thinking": "private"},
                    {
                        "type": "tool_use",
                        "name": "submit",
                        "input": {"decision": "allow"},
                    },
                ]

        session = Session()
        tool = {"type": "function", "function": {"name": "submit"}}
        with patch.object(agentmain, "resolve_session", return_value=session) as resolve:
            result = agent.model_tool_call("gate prompt", tool, require_tool=True)

        resolve.assert_called_once_with("native_oai_config")
        self.assertEqual(session.messages, [{"role": "user", "content": "gate prompt"}])
        self.assertEqual(session.tools, [tool])
        self.assertEqual(
            session.tool_choice,
            {"type": "function", "function": {"name": "submit"}},
        )
        self.assertEqual(session.temperature, 0)
        self.assertEqual(result, [{"type": "tool_use", "name": "submit", "input": {"decision": "allow"}}])
        self.assertEqual(agent.llmclient.backend.history, ["unchanged"])

    def test_openai_transport_sends_required_tool_choice(self):
        captured = {}
        session = SimpleNamespace(
            model="test-model",
            api_mode="chat_completions",
            temperature=0,
            api_key="test-key",
            user_agent="test-agent",
            system="",
            stream=False,
            reasoning_effort=None,
            max_tokens=None,
            tools=[{"type": "function", "function": {"name": "submit"}}],
            tool_choice={"type": "function", "function": {"name": "submit"}},
            service_tier=None,
            api_base="https://example.invalid/v1",
        )

        def fake_retry(_session, _url, _headers, payload, _parse):
            captured["payload"] = payload
            if False:
                yield None
            return []

        with patch.object(llmcore, "_stream_with_retry", fake_retry):
            list(llmcore._openai_stream(session, [{"role": "user", "content": "gate"}]))

        assert captured["payload"]["tool_choice"] == session.tool_choice

    def test_execute_task_owns_history_state_loop_and_cleanup(self):
        captured = {}

        def loop(client, system, query, handler, schema, **kwargs):
            captured.update(system=system, query=query, handler=handler, kwargs=kwargs)
            handler.history_info.append("[Agent] done")
            handler.terminal_text = "final answer"
            yield {"turn": 1}
            yield "chunk"
            return {"result": "CURRENT_TASK_DONE"}

        agent = fake_agent()
        previous = Handler(agent, [], ".")
        previous.working = {"key_info": "keep", "passed_sessions": 2}
        agent.handler = previous
        chunks = []
        with patch.object(agentmain, "agent_runner_loop", loop), patch.object(
            agentmain, "get_system_prompt", return_value="base"
        ):
            result = agent.execute_task(
                "model input",
                handler_class=Handler,
                cwd="workspace",
                extra_system_prompt="runtime",
                history_content="actual user text",
                max_turns=7,
                initial_user_content="model input",
                on_chunk=chunks.append,
            )

        self.assertEqual(agent.history, ["[USER]: actual user text", "[Agent] done"])
        self.assertEqual(captured["query"], "model input")
        self.assertIn("runtime", captured["system"])
        self.assertIn(f"Current tool cwd: {Path('workspace').resolve()} (./)", captured["system"])
        self.assertEqual(agent.handler.cwd, str(Path("workspace").resolve()))
        self.assertEqual(captured["kwargs"]["max_turns"], 7)
        self.assertEqual(captured["kwargs"]["initial_user_content"], "model input")
        self.assertEqual(agent.handler.working["passed_sessions"], 3)
        self.assertEqual(chunks, [{"turn": 1}, "chunk"])
        self.assertTrue(agent.handler.finished)
        self.assertEqual(agent.handler.code_stop_signal, [1])
        self.assertFalse(agent.is_running)
        self.assertFalse(agent.stop_sig)
        self.assertEqual(result["loop_result"], {"result": "CURRENT_TASK_DONE"})
        self.assertEqual(result["outcome"], "CURRENT_TASK_DONE")
        self.assertEqual(result["terminal_text"], "final answer")
        self.assertIsNone(result["interrupt"])
        self.assertFalse(result["aborted"])

    def test_abort_stops_the_native_loop_and_still_finishes_handler(self):
        agent = fake_agent()

        class AbortingHandler(Handler):
            def __init__(self, parent, history, cwd):
                super().__init__(parent, history, cwd)
                parent.abort()

        def loop(*_args, **_kwargs):
            raise AssertionError("aborted task entered the model loop")
            yield

        with patch.object(agentmain, "agent_runner_loop", loop), patch.object(
            agentmain, "get_system_prompt", return_value="base"
        ):
            result = agent.execute_task("task", handler_class=AbortingHandler)

        self.assertTrue(result["aborted"])
        self.assertTrue(agent.handler.finished)
        self.assertFalse(agent.is_running)
        self.assertFalse(agent.stop_sig)

    def test_execute_task_exposes_interrupt_without_frontend_state(self):
        agent = fake_agent()
        payload = {
            "status": "INTERRUPT",
            "intent": "HUMAN_INTERVENTION",
            "data": {"question": "continue?", "candidates": ["yes", "no"]},
        }

        def loop(_client, _system, _query, handler, _schema, **_kwargs):
            handler.terminal_text = "continue?\n- yes\n- no"
            if False: yield
            return {"result": "EXITED", "data": payload}

        with patch.object(agentmain, "agent_runner_loop", loop), patch.object(
            agentmain, "get_system_prompt", return_value="base"
        ):
            result = agent.execute_task("task", handler_class=Handler)

        self.assertEqual(result["outcome"], "EXITED")
        self.assertEqual(result["terminal_text"], "continue?\n- yes\n- no")
        self.assertEqual(result["interrupt"], payload)

    def test_long_input_is_externalized_once_with_an_explicit_history_preview(self):
        agent = fake_agent()
        captured = {}

        def loop(_client, _system, query, _handler, _schema, **kwargs):
            captured.update(query=query, initial=kwargs["initial_user_content"])
            if False: yield
            return {"result": "CURRENT_TASK_DONE"}

        query = "x" * 3000
        with tempfile.TemporaryDirectory() as cwd, patch.object(
            agentmain, "agent_runner_loop", loop
        ), patch.object(agentmain, "get_system_prompt", return_value="base"):
            agent.execute_task(query, handler_class=Handler, cwd=cwd)
            path = Path(captured["query"].removeprefix("Long user prompt saved to ").removesuffix(". Read and execute."))
            self.assertEqual(path.read_text(encoding="utf-8"), query)

        self.assertEqual(captured["query"], captured["initial"])
        self.assertTrue(agent.history[0].startswith("[USER] [preview; original 3000 chars]:"))
        folded = agentmain.GenericAgentHandler(agent, [], ".")._fold_earlier(
            [agent.history[0], "[Agent] done"]
        )
        self.assertTrue(folded.startswith("[USER] [preview;"))


if __name__ == "__main__":
    unittest.main()

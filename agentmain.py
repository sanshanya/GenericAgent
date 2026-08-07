import os, sys, threading, queue, time, json, re, random, locale, glob, shutil
os.environ.setdefault('GA_LANG', 'zh' if any(k in (locale.getlocale()[0] or '').lower() for k in ('zh', 'chinese')) else 'en')
if sys.stdout is None: sys.stdout = open(os.devnull, "w")
elif hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(errors='replace')
if sys.stderr is None: sys.stderr = open(os.devnull, "w")
elif hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(errors='replace')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llmcore import reload_mykeys, ToolClient, MixinSession, NativeToolClient, NativeClaudeSession, NativeOAISession, resolve_client, resolve_session, fast_ask
from agent_loop import agent_runner_loop
try:
    from plugins.hooks import discover_and_load; discover_and_load()
except Exception: pass
from ga import GenericAgentHandler, smart_format, get_global_memory, format_error, consume_file, memory_root

script_dir = os.path.dirname(os.path.abspath(__file__))
BANNED_TOOLS = (['ask_user', 'start_long_term_update'] if '--no-user-tools' in sys.argv else [])
def load_tool_schema(suffix=''):
    global TOOLS_SCHEMA
    TS = open(os.path.join(script_dir, f'assets/tools_schema{suffix}.json'), 'r', encoding='utf-8').read()
    TOOLS_SCHEMA = json.loads(TS if os.name == 'nt' else TS.replace('powershell', 'bash'))
    TOOLS_SCHEMA = [t for t in TOOLS_SCHEMA if t.get('function', {}).get('name') not in BANNED_TOOLS]
load_tool_schema()

lang_suffix = '_en' if os.environ.get('GA_LANG', '') == 'en' else ''
mem_dir = memory_root
bundled_memory = os.path.join(script_dir, 'memory')
if os.path.abspath(mem_dir) != os.path.abspath(bundled_memory):
    for source_dir, _, files in os.walk(bundled_memory):
        target_dir = os.path.join(mem_dir, os.path.relpath(source_dir, bundled_memory))
        os.makedirs(target_dir, exist_ok=True)
        for name in files:
            source, target = os.path.join(source_dir, name), os.path.join(target_dir, name)
            if not os.path.exists(target): shutil.copy2(source, target)
os.makedirs(mem_dir, exist_ok=True)
mem_txt = os.path.join(mem_dir, 'global_mem.txt')
if not os.path.exists(mem_txt): open(mem_txt, 'w', encoding='utf-8').write('# [Global Memory - L2]\n')
mem_insight = os.path.join(mem_dir, 'global_mem_insight.txt')
if not os.path.exists(mem_insight):
    t = os.path.join(script_dir, f'assets/global_mem_insight_template{lang_suffix}.txt')
    open(mem_insight, 'w', encoding='utf-8').write(open(t, encoding='utf-8').read() if os.path.exists(t) else '')

def get_system_prompt():
    with open(os.path.join(script_dir, f'assets/sys_prompt{lang_suffix}.txt'), 'r', encoding='utf-8') as f: prompt = f.read()
    prompt += f"\nToday: {time.strftime('%Y-%m-%d %a')}\n"
    prompt += get_global_memory()
    return prompt

# SDK:
# agent = GenericAgent(); threading.Thread(target=agent.run, daemon=True).start()
# output1_queue = agent.put_task(prompt1)
# output2_queue = agent.put_task(prompt2)
class GenericAgent:
    def __init__(self):
        os.makedirs(os.path.join(script_dir, 'temp'), exist_ok=True)
        self.lock = threading.Lock()
        self._intervention_lock = threading.Lock()
        self._intervention_pending = []
        self._intervention_replay = []
        self._turn_end_hooks = {}
        self.task_dir = None
        self.history = []; self.handler = None; self.all_outputs = []
        self.task_queue = queue.Queue() 
        self.is_running = False; self.stop_sig = False; self.llm_no = 0;
        # Output queue of the task currently executing (None when idle). Lets a UI that
        # lost its own handle (page refresh, second client) re-attach to the live task.
        self._current_queue = None  
        self.inc_out = False; self.verbose = True
        self.peer_hint = True
        self.force_non_stream = False
        logid = f'{(time.time_ns() + random.randrange(1_000_000)) % 1_000_000:06d}'
        self.log_path = os.path.join(script_dir, f'temp/model_responses/model_responses_{logid}.txt')
        self.llmclient = None
        self.load_llm_sessions()
        self.extra_sys_prompts = []
        self.intervene = self.extrakeyinfo = None

    def inject_intervene(self, text):
        """Deliver user steering input at the next native turn boundary.

        This is the same file seam used by the native TUI. It does not append
        to Agent history or start a second loop. A caller that gets False must
        submit the text as a normal next task instead.
        """
        text = str(text or '').strip()
        if not text:
            return False
        lock = getattr(self, '_intervention_lock', None)
        if lock is None:
            lock = self._intervention_lock = threading.Lock()
        with lock:
            if not getattr(self, 'is_running', False) or not getattr(self, 'task_dir', None):
                return False
            os.makedirs(self.task_dir, exist_ok=True)
            with open(os.path.join(self.task_dir, '_intervene'), 'a', encoding='utf-8') as stream:
                stream.write(text + '\n\n')
            pending = getattr(self, '_intervention_pending', None)
            if pending is None:
                pending = self._intervention_pending = []
            pending.append(text)
            return True

    def _track_intervention_boundary(self, context):
        lock = getattr(self, '_intervention_lock', None)
        if lock is None:
            return
        with lock:
            pending = getattr(self, '_intervention_pending', [])
            if not pending:
                return
            if context.get('exit_reason'):
                replay = getattr(self, '_intervention_replay', None)
                if replay is None:
                    replay = self._intervention_replay = []
                replay.extend(pending)
            pending.clear()

    def load_llm_sessions(self):
        mykeys, changed = reload_mykeys()
        if not changed and hasattr(self, 'llmclients'): return
        try: oldhistory, oldname = self.llmclient.backend.history, self.llmclient.backend.name
        except: oldhistory = oldname = None
        llm_sessions = []
        for k, cfg in mykeys.items():
            if not any(x in k for x in ['api', 'config', 'cookie']): continue
            try:
                if 'mixin' in k: llm_sessions += [{'mixin_cfg': cfg}]
                elif c := resolve_client(k): llm_sessions += [c]
            except: pass
        for i, s in enumerate(llm_sessions):
            if isinstance(s, dict) and 'mixin_cfg' in s:
                try:
                    mixin = MixinSession(llm_sessions, s['mixin_cfg'])
                    if isinstance(mixin._sessions[0], (NativeClaudeSession, NativeOAISession)): llm_sessions[i] = NativeToolClient(mixin)
                    else: llm_sessions[i] = ToolClient(mixin)
                except Exception as e: print(f'\n\n\n[ERROR] Failed to init MixinSession with cfg {s["mixin_cfg"]}: {e}!!!\n\n')
        self.llmclients = llm_sessions
        if not self.llmclients: return
        names = [c.backend.name if not isinstance(c, dict) else f'BADMIXIN_{i}' for i, c in enumerate(self.llmclients)]
        if oldname in names: self.llm_no = names.index(oldname)
        self.llmclient = self.llmclients[self.llm_no%len(self.llmclients)]
        if oldhistory: self.llmclient.backend.history = oldhistory
    
    def next_llm(self, n=-1):
        self.load_llm_sessions()
        if not self.llmclients: return
        self.llm_no = ((self.llm_no + 1) if n < 0 else n) % len(self.llmclients)
        lastc = self.llmclient
        self.llmclient = self.llmclients[self.llm_no]
        try: self.llmclient.backend.history = lastc.backend.history
        except: raise Exception('[ERROR] BAD Mixin config: Check your mykey.py')
        self.llmclient.last_tools = ''
        load_tool_schema()
    def list_llms(self): 
        self.load_llm_sessions()
        return [(i, self.get_llm_name(b), i == self.llm_no) for i, b in enumerate(self.llmclients)]
    def get_llm_name(self, b=None, model=False):
        b = self.llmclient if b is None else b
        if isinstance(b, dict): return 'BADCONFIG_MIXIN'
        if model: return b.backend.model.lower()
        return f"{type(b.backend).__name__.replace('Session', '')}/{b.backend.name}"
    def get_ctx_multiplier(self): return getattr(self.llmclient.backend, 'maxlen_multiplier', 1.0)

    def model_call(self, prompt, temperature=0):
        config_name = getattr(self.llmclient.backend, 'config_name', '')
        if not config_name: raise RuntimeError('Current GA model has no reusable config name')
        return fast_ask(prompt, config_name, temperature=temperature)

    def model_tool_call(self, prompt, tool, temperature=0, require_tool=False):
        config_name = getattr(self.llmclient.backend, 'config_name', '')
        if not config_name: raise RuntimeError('Current GA model has no reusable config name')
        session = resolve_session(config_name)
        if not session: raise RuntimeError(f"Model config '{config_name}' cannot create a session")
        session.temperature = temperature
        session.tools = [tool]
        tool_name = (tool.get("function") or {}).get("name")
        if require_tool and not tool_name:
            raise ValueError("required tool must be a named function")
        session.tool_choice = (
            {"type": "function", "function": {"name": tool_name}}
            if require_tool else None
        )
        generator = session.raw_ask([{"role": "user", "content": prompt}])
        try:
            while True: next(generator)
        except StopIteration as stop:
            blocks = stop.value or []
        return [block for block in blocks if block.get("type") == "tool_use"]

    def abort(self):
        if not self.is_running: return
        print('Abort current task...')
        self.stop_sig = True
        if self.handler is not None: self.handler.code_stop_signal.append(1)
        for sess in getattr(self.llmclient.backend, '_sessions', [self.llmclient.backend]):
            sess.should_stop = lambda: self.stop_sig  # live read; cleared by run()'s finally
            try: sess.active_response.close()
            except Exception: pass
            
    def put_task(self, query, source="user", images=None):
        display_queue = queue.Queue()
        self.task_queue.put({"query": query, "source": source, "images": images or [], "output": display_queue})
        return display_queue

    # i know it is dangerous, but raw_query is dangerous enough it doesn't enlarge
    def _handle_slash_cmd(self, raw_query, display_queue):
        if not raw_query.startswith('/'): return raw_query
        if _sm := re.match(r'/session\.(\w+)=(.*)', raw_query.strip()):
            k, v = _sm.group(1), _sm.group(2)
            vfile = os.path.join(script_dir, 'temp', v)
            if os.path.isfile(vfile): v = open(vfile, encoding='utf-8').read().strip()
            try: v = json.loads(v)  # cover number parsing
            except (json.JSONDecodeError, ValueError): pass
            setattr(self.llmclient.backend, k, v)
            display_queue.put({'done': smart_format(f"✅ session.{k} = {repr(v)}", max_str_len=500), 'source': 'system'})
            return None
        if raw_query.strip() == '/resume':
            return r'帮我看看最近有哪些会话可以恢复。读model_responses/目录，按修改时间取最近10个文件，从每个文件里找最后一个<history>...</history>块，用一句话总结每个会话在聊什么，列表给我选。注意读文件后要把字面的\n替换成真换行才能正确匹配。'
        return raw_query

    @staticmethod
    def _prepare_task_input(query, history_content, cwd):
        original = history_content if history_content is not None else query
        compact = smart_format(original.replace('\n', ' '), max_str_len=200)
        history = (
            f"[USER] [preview; original {len(original)} chars]: {compact}"
            if compact != original.replace('\n', ' ')
            else f"[USER]: {compact}"
        )
        if len(query) <= 2000: return query, history
        os.makedirs(cwd, exist_ok=True)
        task_file = os.path.join(cwd, f'user_prompt_{os.getpid()}_{time.time_ns()}.md')
        with open(task_file, 'w', encoding='utf-8') as f: f.write(query)
        return f'Long user prompt saved to {task_file}. Read and execute.', history

    def execute_task(self, query, *, handler_class=GenericAgentHandler, cwd=None,
                     extra_system_prompt='', history_content=None, max_turns=180,
                     initial_user_content=None, yield_info=True, on_chunk=None):
        """Execute one task while preserving GenericAgent's native cognitive state."""
        self.is_running, self.stop_sig = True, False
        handler = gen = result = None
        hooks = getattr(self, '_turn_end_hooks', None)
        if hooks is None:
            hooks = self._turn_end_hooks = {}
        hooks.setdefault('_generic_agent_intervention', self._track_intervention_boundary)
        try:
            cwd = os.path.abspath(cwd or os.path.join(script_dir, 'temp'))
            prepared, history = self._prepare_task_input(query, history_content, cwd)
            self.history.append(history)
            if initial_user_content is None or initial_user_content == query:
                initial_user_content = prepared
            sys_prompt = get_system_prompt()
            sys_prompt += f'\nCurrent tool cwd: {cwd} (./)\n'
            if extra_system_prompt: sys_prompt += '\n' + extra_system_prompt
            sys_prompt += '\n'.join(self.extra_sys_prompts)
            sys_prompt += getattr(self.llmclient.backend, 'extra_sys_prompt', '')
            if self.peer_hint:
                sys_prompt += f"\n[Peer] 用户提及其他会话/后台任务状态时: temp/model_responses/ (只找近期修改的文件尾部)\n"
            handler = handler_class(self, self.history, cwd)
            if getattr(self, 'no_print', False): handler.print = lambda *a, **k: None
            if self.handler and 'key_info' in self.handler.working:
                ki = re.sub(r'\n\[SYSTEM\] 此为.*?工作记忆[。\n]*', '', self.handler.working['key_info'])
                handler.working['key_info'] = ki
                handler.working['passed_sessions'] = ps = self.handler.working.get('passed_sessions', 0) + 1
                if ps > 0: handler.working['key_info'] += f'\n[SYSTEM] 此为 {ps} 个对话前设置的key_info，若已在新任务，先更新或清除工作记忆。\n'
            self.handler = handler
            self.llmclient.log_path = self.log_path
            if self.force_non_stream:
                self.llmclient.backend.stream = False
                self.llmclient.backend.read_timeout = max(self.llmclient.backend.read_timeout, 1200)
            gen = agent_runner_loop(
                self.llmclient, sys_prompt, prepared, handler, TOOLS_SCHEMA,
                max_turns=max_turns, verbose=self.verbose,
                initial_user_content=initial_user_content, yield_info=yield_info,
            )
            loop_result = None; aborted = self.stop_sig
            while not aborted:
                try: chunk = next(gen)
                except StopIteration as stopped:
                    loop_result = stopped.value
                    break
                if on_chunk: on_chunk(chunk)
                aborted = self.stop_sig
            self.history = handler.history_info
            outcome = loop_result.get('result', '') if isinstance(loop_result, dict) else ''
            interrupt = loop_result.get('data') if outcome == 'EXITED' else None
            result = {
                'loop_result': loop_result,
                'outcome': outcome,
                'terminal_text': getattr(handler, 'terminal_text', ''),
                'interrupt': interrupt,
                'aborted': aborted,
                'intervention_replay': [],
            }
            return result
        finally:
            try:
                if gen is not None: gen.close()
            finally:
                try:
                    if handler is not None and hasattr(handler, 'finish_task'): handler.finish_task()
                finally:
                    lock = getattr(self, '_intervention_lock', None)
                    if lock is None:
                        lock = self._intervention_lock = threading.Lock()
                    with lock:
                        pending = getattr(self, '_intervention_pending', [])
                        if pending:
                            replay = getattr(self, '_intervention_replay', None)
                            if replay is None:
                                replay = self._intervention_replay = []
                            replay.extend(pending)
                            pending.clear()
                        if result is not None:
                            replay = getattr(self, '_intervention_replay', [])
                            result['intervention_replay'] = list(replay)
                            replay.clear()
                        self.is_running = self.stop_sig = False
                    if handler is not None: handler.code_stop_signal.append(1)

    def run(self):
        while True:
            task = self.task_queue.get()
            if isinstance(task, str): break
            raw_query, source, display_queue = task["query"], task["source"], task["output"]
            raw_query = self._handle_slash_cmd(raw_query, display_queue)
            if raw_query is None:
                self.task_queue.task_done(); continue
            self._current_queue = display_queue
            self.all_outputs.append({"input": raw_query, "outputs": []})
            if len(self.all_outputs) > 10000: self.all_outputs = self.all_outputs[-5000:]
            try:
                full_resp = ""; last_pos = 0; curr_turn = 0; turn_resps = self.all_outputs[-1]["outputs"]
                def on_chunk(chunk):
                    nonlocal full_resp, last_pos, curr_turn
                    if consume_file(self.task_dir, '_stop'): self.abort()

                    if isinstance(chunk, dict) and 'turn' in chunk: 
                        curr_turn = chunk['turn']; turn_resps.append(''); return
                    full_resp += chunk;  turn_resps[-1] += chunk
                    if len(full_resp) - last_pos > 30 or 'LLM Running' in chunk:
                        display_queue.put({'next': full_resp[last_pos:] if self.inc_out else full_resp, 
                                           'source': source, 'turn': curr_turn, 'outputs': turn_resps[-2:]})
                        last_pos = len(full_resp)
                execution = self.execute_task(raw_query, yield_info=True, on_chunk=on_chunk)
                if self.inc_out and last_pos < len(full_resp):
                    display_queue.put({'next': full_resp[last_pos:], 'source': source,
                                    'turn': curr_turn, 'outputs': turn_resps[-2:]})
                display_queue.put({'done': full_resp, 'source': source, 'turn': curr_turn, 'outputs': turn_resps.copy()})
                if execution['aborted']: print('User aborted the task.')
            except Exception as e:
                print(f"Backend Error: {format_error(e)}")
                display_queue.put({'done': full_resp + f'\n```\n{format_error(e)}\n```', 'source': source, 'turn': curr_turn, 'outputs': turn_resps.copy()})
            finally:

                self.task_queue.task_done()

GeneraticAgent = GenericAgent

if __name__ == '__main__':
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', metavar='IODIR', help='一次性任务模式，先看subagent.md')
    parser.add_argument('--func', metavar='PROMPT_FILE', help='纯函数模式：读prompt文件→结果写prompt.out.txt→退出')
    parser.add_argument('--reflect', metavar='SCRIPT', help='反射模式：加载监控脚本，check()触发时发任务')
    parser.add_argument('--input', help='prompt')
    parser.add_argument('--history', help='history json file')
    parser.add_argument('--llm_no', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--nobg', action='store_true')
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--no-user-tools', action='store_true')
    args, _unknown = parser.parse_known_args()
    _extra_args = dict(zip([k.lstrip('-') for k in _unknown[::2]], _unknown[1::2])) if _unknown else {}

    if (args.func or args.task) and not args.nobg:
        import subprocess, platform
        cmd = [sys.executable, os.path.abspath(__file__)] + [a for a in sys.argv[1:]] + ['--nobg']
        if args.task:
            d = os.path.join(script_dir, f'temp/{args.task}'); os.makedirs(d, exist_ok=True)
            out = open(os.path.join(d, 'stdout.log'), 'w', encoding='utf-8')
            err = open(os.path.join(d, 'stderr.log'), 'w', encoding='utf-8')
        else: out, err = subprocess.DEVNULL, subprocess.DEVNULL
        p = subprocess.Popen(cmd, cwd=script_dir,
            creationflags=0x08000000 if platform.system() == 'Windows' else 0,
            stdout=out, stderr=err)
        print('PID:', p.pid); sys.exit(0)

    agent = GenericAgent()
    if args.nolog: agent.log_path = False
    agent.next_llm(args.llm_no)
    agent.verbose = args.verbose
    threading.Thread(target=agent.run, daemon=True).start()

    histfile = args.history
    if args.task:
        agent.task_dir = d = os.path.join(script_dir, f'temp/{args.task}'); nround = ''
        infile = os.path.join(d, 'input.txt'); outfile = f'{d}/output{nround}.txt'
        if args.input:
            os.makedirs(d, exist_ok=True)
            [os.remove(f) for f in glob.glob(os.path.join(d, 'output*.txt'))]
            with open(infile, 'w', encoding='utf-8') as f: f.write(args.input)
        histfile = histfile or os.path.join(d, '_history.json')
    elif args.func:
        infile = args.func; outfile = os.path.splitext(args.func)[0] + '.out.txt'

    if histfile and os.path.isfile(histfile): agent.llmclient.backend.history = json.loads(open(histfile, encoding='utf-8').read())

    if args.func or args.task:
        agent.peer_hint = False
        with open(infile, encoding='utf-8') as f: raw = f.read()
        while True:
            dq = agent.put_task(raw, source='func' if args.func else 'task')
            while 'done' not in (item := dq.get(timeout=2200)):
                if 'next' in item:
                    with open(outfile, 'w', encoding='utf-8') as f: f.write(item.get('next', ''))
            with open(outfile, 'w', encoding='utf-8') as f: f.write(item['done'] + '\n\n[ROUND END]\n')
            if not args.task: break
            consume_file(d, '_stop')  # 已经成功停下来了，避免打断下次reply
            for _ in range(300):  # 等reply.txt，10分钟超时
                time.sleep(2)
                if (raw := consume_file(d, 'reply.txt')): break
            else: break
            nround = nround + 1 if isinstance(nround, int) else 1
            outfile = f'{d}/output{nround}.txt'
    elif args.reflect:
        agent.peer_hint = False
        import importlib.util
        spec = importlib.util.spec_from_file_location('reflect_script', args.reflect)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        if hasattr(mod, 'init'): mod.init(_extra_args)
        _mt = os.path.getmtime(args.reflect)
        print(f'[Reflect] loaded {args.reflect}' + (f' args={_extra_args}' if _extra_args else ''))
        while True:
            if os.path.getmtime(args.reflect) != _mt:
                try:
                    spec.loader.exec_module(mod); _mt = os.path.getmtime(args.reflect)
                    if hasattr(mod, 'init'): mod.init(_extra_args)
                    print('[Reflect] reloaded')
                except Exception as e: print(f'[Reflect] reload error: {e}')
            try: task = mod.check()
            except Exception as e: 
                print(f'[Reflect] check() error: {e}'); task = None
            if task and task == '/exit': break
            if task:
                print(f'[Reflect] triggered: {task[:80]}')
                dq = agent.put_task(task, source='reflect')
                try:
                    while 'done' not in (item := dq.get(timeout=2200)): pass
                    result = item['done']
                    print(result)
                except Exception as e:
                    if getattr(mod, 'ONCE', False): raise
                    print(f'[Reflect] drain error: {e}'); result = f'[ERROR] {e}'
                log_dir = os.path.join(script_dir, 'temp/reflect_logs'); os.makedirs(log_dir, exist_ok=True)
                script_name = os.path.splitext(os.path.basename(args.reflect))[0]
                open(os.path.join(log_dir, f'{script_name}_{datetime.now():%Y-%m-%d}.log'), 'a', encoding='utf-8').write(f'[{datetime.now():%m-%d %H:%M}]\n{result}\n\n')
                if (on_done := getattr(mod, 'on_done', None)):
                    try: on_done(result)
                    except Exception as e: print(f'[Reflect] on_done error: {e}')
                if getattr(mod, 'ONCE', False): print('[Reflect] ONCE=True, exiting.'); break
            time.sleep(getattr(mod, 'INTERVAL', 5))
    else:
        try: import readline
        except Exception: pass
        agent.inc_out = True
        if sys.stdout.isatty():
            try: model = agent.get_llm_name(model=True) or '?'
            except Exception: model = '?'
            try:
                sys.stdout.write(f'\x1b[92m✦\x1b[0m \x1b[1mGenericAgent\x1b[0m '
                                 f'\x1b[90m· cli · model:\x1b[0m {model}\n')
                sys.stdout.flush()
            except Exception: pass
        while True:
            q = input('> ').strip()
            if not q: continue
            try:
                dq = agent.put_task(q, source='user')
                while True:
                    item = dq.get()
                    if 'next' in item: print(item['next'], end='', flush=True)
                    if 'done' in item: print(); break
            except KeyboardInterrupt:
                agent.abort(); print('\n[Interrupted]')

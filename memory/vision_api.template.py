import base64, requests, sys, os
from io import BytesIO
from pathlib import Path

# ============ 用户配置区（从 template 拷贝后只需改这里）============
# GA自举提示：
# mykey.py配置变量名不固定；默认值失败时，枚举候选变量名，并用最小vision请求实验探测哪组配置可用。
# 只打印变量名/字段名/model/apibase域名路径/HTTP状态码/错误类型，禁止打印完整dict和apikey/token。
# 常见格式：
#   xxx_config = {"apibase": "https://.../v1", "apikey": "...", "model": "...", "proxy": None}
# 不同中转的 apibase/endpoint 可能不同，按实际状态码和响应结构修正。
CLAUDE_CONFIG_KEY = 'claude_config141'   # mykey.py 中 Claude 配置的变量名
OPENAI_CONFIG_KEY = 'oai_config1'        # mykey.py 中 OpenAI 配置的变量名
# 可选：GA_VISION_LLM_NO=1 时按 GenericAgent 的第 1 个 provider 取 OpenAI 兼容配置；未设置时继续使用 OPENAI_CONFIG_KEY。
MODELSCOPE_API_KEY = ''                  # 直接填你的 ModelScope token
DEFAULT_BACKEND = 'claude'               # 默认后端: 'claude' / 'openai' / 'modelscope'
# =================================================================

MODELSCOPE_API_BASE = 'https://api-inference.modelscope.cn'
MODELSCOPE_MODEL = 'Qwen/Qwen3-VL-235B-A22B-Instruct'

_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_DIR, '..'), os.path.join(_DIR, '../..')]:
    if _p not in sys.path: sys.path.insert(0, _p)

def ask_vision(image_input, prompt="详细描述这张图片的内容", timeout=60, max_pixels=1440000, backend=DEFAULT_BACKEND):
    try:
        b64 = _prepare_image(image_input, max_pixels)
    except Exception as e:
        return f"Error: 图片处理失败 - {type(e).__name__}: {e}"
    try:
        if backend == 'claude':
            return _call_claude(b64, prompt, timeout)
        elif backend == 'openai':
            cfg = _openai_config()
            return _call_openai_compat(
                b64, prompt, timeout,
                apibase=cfg['apibase'], apikey=cfg['apikey'], model=cfg['model'], proxy=cfg.get('proxy')
            )
        elif backend == 'modelscope':
            return _call_openai_compat(
                b64, prompt, timeout,
                apibase=MODELSCOPE_API_BASE, apikey=MODELSCOPE_API_KEY, model=MODELSCOPE_MODEL, proxy=None
            )
        else: return f"Error: 未知backend '{backend}'，可选: claude, openai, modelscope"
    except requests.exceptions.Timeout:
        return f"Error: 请求超时 (>{timeout}s)"
    except requests.exceptions.RequestException as e:
        return f"Error: API请求失败 - {type(e).__name__}: {e}"
    except (KeyError, ValueError) as e:
        return f"Error: 响应解析失败 - {e}"

# ===================== 以下为内部实现 =====================

def _prepare_image(image_input, max_pixels=1440000):
    """加载+缩放+base64编码，返回b64字符串"""
    from PIL import Image
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    else:
        raise TypeError(f"image_input 必须是文件路径或PIL Image，实际: {type(image_input).__name__}")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"  resized image: {w}x{h} -> {new_w}x{new_h}")
    if img.mode in ('RGBA', 'LA', 'P'):
        rgb = Image.new('RGB', img.size, (255, 255, 255))
        rgb.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = rgb
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=80, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    print(f"  encoded image: {len(buf.getvalue())/1024:.1f}KB")
    return b64

def _load_config():
    _ensure_ga_root()
    import mykey
    return mykey

def _ensure_ga_root():
    ga_root = os.getenv('GA_ROOT', '').strip()
    if not ga_root:
        return
    ga_root = os.path.abspath(ga_root)
    if ga_root not in sys.path:
        sys.path.insert(0, ga_root)

def _openai_config():
    selected = os.getenv('GA_VISION_LLM_NO', '').strip()
    if not selected:
        return getattr(_load_config(), OPENAI_CONFIG_KEY)
    try:
        llm_no = int(selected)
    except ValueError as e:
        raise ValueError(f"GA_VISION_LLM_NO must be an integer, got {selected!r}") from e
    _ensure_ga_root()
    from agentmain import GenericAgent
    agent = GenericAgent()
    if not 0 <= llm_no < len(agent.llmclients):
        raise ValueError(f"GA_VISION_LLM_NO={llm_no} outside configured range")
    backend = agent.llmclients[llm_no].backend
    if not all(hasattr(backend, key) for key in ('api_base', 'api_key', 'model')):
        raise ValueError(f"GA_VISION_LLM_NO={llm_no} is not OpenAI-compatible")
    proxies = getattr(backend, 'proxies', None)
    proxy = proxies.get('https') or proxies.get('http') if isinstance(proxies, dict) else None
    return {'apibase': backend.api_base, 'apikey': backend.api_key, 'model': backend.model, 'proxy': proxy}

def _call_claude(b64, prompt, timeout, max_tokens=1024):
    mk = _load_config()
    cfg = getattr(mk, CLAUDE_CONFIG_KEY)
    resp = requests.post(
        cfg['apibase'] + '/v1/messages',   # endpoint按中转实际情况改：有的apibase已含/v1，或路径不同
        json={'model': cfg['model'], 'max_tokens': max_tokens, 'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': b64}},
                {'type': 'text', 'text': prompt}
            ]
        }]},
        headers={'x-api-key': cfg['apikey'], 'anthropic-version': '2023-06-01', 'content-type': 'application/json'},
        timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()['content'][0]['text']

def _call_openai_compat(b64, prompt, timeout, *, apibase, apikey, model, proxy=None):
    proxies = {'https': proxy, 'http': proxy} if proxy else None
    endpoint = apibase.rstrip('/')
    if not endpoint.endswith('chat/completions'):
        endpoint += '/chat/completions' if endpoint.rsplit('/', 1)[-1].startswith('v') else '/v1/chat/completions'
    resp = requests.post(
        endpoint,
        json={'model': model, 'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}
            ]
        }]},
        headers={'Authorization': f"Bearer {apikey}", 'Content-Type': 'application/json'},
        proxies=proxies, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']

if __name__ == '__main__':
    pass

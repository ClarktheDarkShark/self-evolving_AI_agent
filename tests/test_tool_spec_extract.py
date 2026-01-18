import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_controller_module():
    class _DummyRouter:
        def post(self, *args, **kwargs):
            def deco(fn):
                return fn

            return deco

    sys.modules.setdefault("fastapi", types.SimpleNamespace(APIRouter=_DummyRouter))
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    path = repo_root / "src/self_evolving_agent/controller.py"
    pkg_name = "src.self_evolving_agent"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(path.parent)]
    sys.modules.setdefault(pkg_name, pkg)

    reg_path = path.parent / "tool_registry.py"
    reg_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.tool_registry", reg_path, submodule_search_locations=[str(path.parent)]
    )
    reg_module = importlib.util.module_from_spec(reg_spec)
    assert reg_spec and reg_spec.loader
    reg_spec.loader.exec_module(reg_module)
    sys.modules[f"{pkg_name}.tool_registry"] = reg_module

    code = path.read_text(encoding="utf-8")
    module_name = f"{pkg_name}.controller"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = pkg_name
    sys.modules[module_name] = module
    exec(compile(code, str(path), "exec"), module.__dict__)
    return module


def _minimal_spec_dict() -> dict:
    return {
        "name": "spec_test",
        "description": "Test tool",
        "signature": "run(payload: dict) -> dict",
        "tool_type": "utility",
        "tool_category": "validator",
        "input_schema": {
            "type": "object",
            "required": ["payload"],
            "properties": {
                "payload": {"type": "object", "required": [], "properties": {}}
            },
        },
        "capabilities": ["ok"],
        "code_lines": [
            '"""Doc"""',
            "def run(payload: dict) -> dict:",
            "    return {}",
        ],
    }


def test_extract_tool_spec_leading_prose_wrapper():
    ctrl_mod = _load_controller_module()
    ctrl = object.__new__(ctrl_mod.SelfEvolvingController)
    spec = _minimal_spec_dict()
    raw = "Here is the tool:\\n" + json.dumps({"content": json.dumps(spec)}) + "\\nThanks."
    parsed = ctrl.extract_tool_spec(raw, None)
    assert all(k in parsed for k in ctrl._required_tool_spec_keys())
    assert isinstance(parsed["code_lines"], list)
    assert all(isinstance(x, str) for x in parsed["code_lines"])


def test_extract_tool_spec_pure_json():
    ctrl_mod = _load_controller_module()
    ctrl = object.__new__(ctrl_mod.SelfEvolvingController)
    spec = _minimal_spec_dict()
    raw = json.dumps(spec)
    parsed = ctrl.extract_tool_spec(raw, None)
    assert all(k in parsed for k in ctrl._required_tool_spec_keys())
    assert isinstance(parsed["code_lines"], list)


def test_extract_tool_spec_wrapper_only():
    ctrl_mod = _load_controller_module()
    ctrl = object.__new__(ctrl_mod.SelfEvolvingController)
    spec = _minimal_spec_dict()
    raw = json.dumps({"content": json.dumps(spec)})
    parsed = ctrl.extract_tool_spec(raw, None)
    assert all(k in parsed for k in ctrl._required_tool_spec_keys())


def test_extract_tool_spec_tool_calls_args():
    ctrl_mod = _load_controller_module()
    ctrl = object.__new__(ctrl_mod.SelfEvolvingController)
    spec = _minimal_spec_dict()
    response_obj = {"tool_calls": [{"function": {"arguments": json.dumps(spec)}}]}
    parsed = ctrl.extract_tool_spec("", response_obj)
    assert all(k in parsed for k in ctrl._required_tool_spec_keys())
    assert isinstance(parsed["code_lines"], list)

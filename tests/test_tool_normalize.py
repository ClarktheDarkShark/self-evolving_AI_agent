import importlib.util
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

    path = Path(__file__).resolve().parents[1] / "src/self_evolving_agent/controller.py"
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


def _load_registry_module():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    path = repo_root / "src/self_evolving_agent/tool_registry.py"
    spec = importlib.util.spec_from_file_location("tool_registry_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_normalize_code_lines_compile():
    ctrl_mod = _load_controller_module()
    ctrl = object.__new__(ctrl_mod.SelfEvolvingController)
    spec = {
        "name": "normalize_compile_test",
        "description": "Test tool",
        "signature": "run(payload: dict) -> dict",
        "tool_type": "utility",
        "tool_category": "validator",
        "input_schema": {
            "type": "object",
            "required": ["payload"],
            "properties": {
                "payload": {
                    "type": "object",
                    "required": ["foo"],
                    "properties": {"foo": {"type": "string"}},
                }
            },
        },
        "capabilities": ["ok"],
        "code_lines": [
            '"""Test tool"""',
            '"def run(payload: dict) -> dict:"',
            '"    \\"\\\"\\\"Return payload\\\"\\\\"\\\""',
            '"    return {\'ok\': payload}"',
        ],
    }

    normalized = ctrl._normalize_tool_spec(spec.copy())
    source = "\n".join(normalized["code_lines"])
    ns: dict = {}
    exec(compile(source, "<t>", "exec"), ns)
    assert callable(ns["run"])
    assert isinstance(ns["run"]({"foo": "x"}), dict)
    assert callable(ns["self_test"])
    assert ns["self_test"]() is True


def test_register_and_invoke_normalized_tool(tmp_path):
    ctrl_mod = _load_controller_module()
    reg_mod = _load_registry_module()
    ctrl = object.__new__(ctrl_mod.SelfEvolvingController)

    spec = {
        "name": "normalize_register_test",
        "description": "Test tool",
        "signature": "run(payload: dict) -> dict",
        "tool_type": "utility",
        "tool_category": "validator",
        "input_schema": {
            "type": "object",
            "required": ["payload"],
            "properties": {
                "payload": {
                    "type": "object",
                    "required": ["foo"],
                    "properties": {"foo": {"type": "string"}},
                }
            },
        },
        "capabilities": ["ok"],
        "code_lines": [
            '"""Test tool"""',
            "\"def run(payload: dict) -> dict:\"",
            "\"    \\\"\\\"\\\"Doc\\\"\\\"\\\"\"",
            "\"    return {'ok': payload}\"",
        ],
    }

    normalized = ctrl._normalize_tool_spec(spec.copy())
    registry = reg_mod.ToolRegistry(str(tmp_path))
    source = "\n".join(normalized["code_lines"])
    metadata = registry.register_tool(
        name=normalized["name"],
        code=source,
        signature=normalized["signature"],
        description=normalized["description"],
        tool_type=normalized["tool_type"],
        tool_category=normalized["tool_category"],
        input_schema=normalized["input_schema"],
        capabilities=normalized["capabilities"],
    )
    assert metadata is not None
    result = registry.invoke_tool(metadata.name, {"foo": "x"})
    assert result.success
    assert isinstance(result.output, dict)

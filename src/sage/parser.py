from __future__ import annotations

import ast
import re
from dataclasses import dataclass

QUERY_START = "###QUERY_START"
QUERY_END = "###QUERY_END"
_ALLOWED_IMPORT_MODULES = frozenset({"SPARQLWrapper", "json"})
_ALLOWED_FROM_IMPORTS = {
    "SPARQLWrapper": frozenset({"SPARQLWrapper", "JSON"}),
}
_FORBIDDEN_CALL_NAMES = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "input",
        "open",
    }
)


class SAGEParserError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedSAGEProgram:
    code: str
    entrypoint: str


class _SAGESecurityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in _ALLOWED_IMPORT_MODULES:
                self.errors.append(f"import_not_allowed:{alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_name = node.module or ""
        allowed_names = _ALLOWED_FROM_IMPORTS.get(module_name)
        if allowed_names is None:
            self.errors.append(f"import_from_not_allowed:{module_name}")
        else:
            for alias in node.names:
                if alias.name not in allowed_names:
                    self.errors.append(
                        f"import_name_not_allowed:{module_name}.{alias.name}"
                    )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.errors.append("global_statements_are_not_allowed")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.errors.append("nonlocal_statements_are_not_allowed")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.errors.append("async_functions_are_not_allowed")
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.errors.append("lambda_expressions_are_not_allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._get_call_name(node.func)
        if call_name in _FORBIDDEN_CALL_NAMES:
            self.errors.append(f"forbidden_call:{call_name}")
        self.generic_visit(node)

    @staticmethod
    def _get_call_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None


def extract_marked_query_code(raw_text: str) -> str:
    if not raw_text:
        raise SAGEParserError("empty_output")

    start_count = raw_text.count(QUERY_START)
    end_count = raw_text.count(QUERY_END)
    if start_count != 1:
        raise SAGEParserError("marker_start_count")
    if end_count != 1:
        raise SAGEParserError("marker_end_count")

    pattern = re.compile(
        rf"{re.escape(QUERY_START)}(?P<code>.*?){re.escape(QUERY_END)}",
        re.DOTALL,
    )
    match = pattern.search(raw_text)
    if match is None:
        raise SAGEParserError("marker_order")

    prefix = raw_text[: match.start()].strip()
    suffix = raw_text[match.end() :].strip()
    if prefix or suffix:
        raise SAGEParserError("marker_extra_text")

    code = match.group("code")
    if code.startswith("\r\n"):
        code = code[2:]
    elif code.startswith("\n"):
        code = code[1:]
    return _normalize_python_code(code)


def parse_sage_program(raw_text: str) -> ParsedSAGEProgram:
    code = extract_marked_query_code(raw_text)
    return parse_sage_code(code)


def parse_sage_code(code: str) -> ParsedSAGEProgram:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SAGEParserError(f"invalid_python:{exc.msg}") from exc

    _validate_module_shape(tree)
    visitor = _SAGESecurityVisitor()
    visitor.visit(tree)
    if visitor.errors:
        raise SAGEParserError(",".join(visitor.errors))

    entrypoint = _get_entrypoint_name(tree)
    return ParsedSAGEProgram(code=code, entrypoint=entrypoint)


def _normalize_python_code(code: str) -> str:
    normalized = code.replace("\r\n", "\n").replace("\r", "\n").rstrip()
    return normalized + "\n"


def _validate_module_shape(tree: ast.Module) -> None:
    allowed_top_level = (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Expr)
    for node in tree.body:
        if not isinstance(node, allowed_top_level):
            raise SAGEParserError(
                f"invalid_top_level_node:{node.__class__.__name__}"
            )
        if isinstance(node, ast.Expr):
            if not (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                raise SAGEParserError("top_level_expression_not_allowed")

    docstring_count = sum(
        1
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )
    if docstring_count > 1:
        raise SAGEParserError("multiple_module_docstrings")

    function_defs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    if len(function_defs) != 1:
        raise SAGEParserError("expected_exactly_one_function")


def extract_and_validate_code(raw_text: str) -> str:
    return parse_sage_program(raw_text).code


def _get_entrypoint_name(tree: ast.Module) -> str:
    function_defs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    if len(function_defs) != 1:
        raise SAGEParserError("expected_exactly_one_function")
    return function_defs[0].name

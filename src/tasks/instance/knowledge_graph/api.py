import json
import logging
import os
import re
from pathlib import Path
from typing import Union, Optional, Sequence, Any
from pydantic import BaseModel
import inspect
from enum import StrEnum

from .utils.logic_form_util import LogicFormUtil
from .utils.sparql_executor import SparqlExecutor


class Variable(BaseModel):
    type: str
    program: str

    def __hash__(self) -> int:
        return hash(self.program)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Variable):
            return self.program == __value.program
        return False

    def __repr__(self) -> str:
        return self.program

    def is_callable(self) -> bool:
        for not_callable_name in ["count", "argmax", "argmin"]:
            if self.program.startswith(f"({not_callable_name.upper()} "):
                return False
        return True



class KnowledgeGraphAPIException(Exception):
    pass


def resolve_ontology_dir(ontology_dir_path: Optional[str | Path]) -> Optional[Path]:
    ontology_path = Path(ontology_dir_path) if ontology_dir_path is not None else None
    tried_paths: list[Path] = []
    required_files = ["vocab.json", "fb_roles"]

    def _is_valid(path: Path) -> bool:
        return all((path / fname).exists() for fname in required_files)

    env_dir = os.getenv("LIFELONG_KG_ONTOLOGY_DIR")
    if env_dir:
        env_path = Path(env_dir).expanduser()
        if _is_valid(env_path):
            return env_path
        tried_paths.append(env_path)

    if ontology_path is not None:
        if _is_valid(ontology_path):
            return ontology_path
        tried_paths.append(ontology_path)

    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "data" / "knowledge_graph" / "ontology",
        repo_root / "data" / "v0121" / "knowledge_graph" / "ontology",
        repo_root / "src" / "tasks" / "instance" / "knowledge_graph" / "ontology",
        repo_root
        / "src"
        / "tasks"
        / "instance"
        / "knowledge_graph"
        / "assets"
        / "ontology",
    ]
    for candidate in candidates:
        if _is_valid(candidate):
            return candidate
        tried_paths.append(candidate)

    repo_id = os.getenv("LIFELONG_KG_ONTOLOGY_REPO")
    subdir = os.getenv("LIFELONG_KG_ONTOLOGY_SUBDIR")
    if repo_id:
        if not subdir:
            subdir = "data/v0121/knowledge_graph/ontology"
        repo_type = os.getenv("LIFELONG_KG_ONTOLOGY_REPO_TYPE", "dataset")
        try:
            from huggingface_hub import hf_hub_download

            local_paths = []
            for fname in required_files:
                local_paths.append(
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        filename=f"{subdir}/{fname}",
                    )
                )
            return Path(local_paths[0]).parent
        except Exception:
            pass

    strict_flag = os.getenv("LIFELONG_KG_ONTOLOGY_STRICT", "0")
    if strict_flag == "1":
        tried_str = "\n".join(str(path) for path in tried_paths)
        raise FileNotFoundError(
            "KG ontology assets not found. Expected vocab.json and fb_roles in one of:\n"
            f"{tried_str}\n"
            "Provide them by either:\n"
            "(1) setting ontology_dir_path in the YAML to a directory containing vocab.json, or\n"
            "(2) setting LIFELONG_KG_ONTOLOGY_REPO/LIFELONG_KG_ONTOLOGY_SUBDIR so the code can auto-download."
        )
    return None


class KnowledgeGraphAPI:
    class ExtremumFunction(StrEnum):
        ARGMAX = "argmax"
        ARGMIN = "argmin"

    def __init__(self, ontology_dir_path: str, sparql_executor: SparqlExecutor):
        logger = logging.getLogger(__name__)
        ontology_dir = resolve_ontology_dir(ontology_dir_path)
        self.ontology_dir = ontology_dir
        logger.warning("KG ontology_dir=%s", ontology_dir)
        if ontology_dir is None:
            logger.warning(
                "KG ontology assets not found; continuing with empty vocab/fb_roles. "
                "For full KG guidance, set ontology_dir_path or LIFELONG_KG_ONTOLOGY_REPO."
            )
            self.attributes = []
            self.relations = []
            self.range_info = {}
        else:
            vocab_path = ontology_dir / "vocab.json"
            roles_path = ontology_dir / "fb_roles"
            logger.warning(
                "KG roles_path=%s exists=%s",
                roles_path,
                roles_path.exists(),
            )
            if not vocab_path.exists() or not roles_path.exists():
                logger.warning(
                    "KG ontology assets missing expected files; continuing with empty vocab/fb_roles. "
                    "Set ontology_dir_path or LIFELONG_KG_ONTOLOGY_REPO for full KG guidance."
                )
                self.attributes = []
                self.relations = []
                self.range_info = {}
            else:
                with open(vocab_path) as f:
                    vocab = json.load(f)
                    self.attributes = vocab.get("attributes", [])
                    self.relations = vocab.get("relations", [])
                self.range_info = {}
                with open(roles_path, "r") as f:
                    for line in f:
                        line = line.replace("\n", "")
                        fields = line.split()
                        if len(fields) >= 3:
                            self.range_info[fields[1]] = fields[2]
            logger.warning(
                "KG range_info size=%d has_cheese=%s",
                len(self.range_info),
                "food.cheese_milk_source.cheeses" in self.range_info,
            )
        self.variable_to_relations_cache: dict[Variable | str, list[str]] = {}
        self.variable_to_attributes_cache: dict[Variable, list[str]] = {}
        self.sparql_executor = sparql_executor

    @staticmethod
    def _ensure_variable(caller_name: str, argument_list: Sequence[Any]) -> None:
        error_message: Optional[str] = None
        for argument_index, argument in enumerate(argument_list):
            if not isinstance(argument, Variable):
                if error_message is None:
                    error_message = f"{caller_name}: "
                else:
                    error_message += ", "
                error_message += f"<<ARGUMENT{argument_index}>> is not a Variable"
        if error_message is not None:
            raise KnowledgeGraphAPIException(error_message)

    def _validate_attribute(
        self, caller_name: str, variable: Variable, attribute: str
    ) -> None:
        if variable not in self.variable_to_attributes_cache.keys():
            raise KnowledgeGraphAPIException(
                f"{caller_name}: "
                f"Use {self.get_attributes.__name__} to get attributes of the Variable <<ARGUMENT0>> first"
            )
        if attribute not in self.variable_to_attributes_cache[variable]:
            raise KnowledgeGraphAPIException(
                f"{caller_name}: <<ARGUMENT1>> is not an attribute of the Variable <<ARGUMENT0>>. "
                f"The attributes of the Variable <<ARGUMENT0>> are: {self.variable_to_attributes_cache[variable]}"
            )

    @staticmethod
    def _validate_variable(caller_name: str, variable_list: Sequence[Variable]) -> None:
        """
        Variable returned by argmax, argmin and count can only be used as final answer.
        """
        error_message: Optional[str] = None
        for variable_index, variable in enumerate(variable_list):
            if variable.is_callable():
                continue
            if error_message is None:
                error_message = f"{caller_name}: "
            else:
                error_message += " "
            # (COUNT (...)) -> COUNT -> count
            # (ARGMAX (...) attribute) -> ARGMAX -> argmax
            api_name = variable.program.split(" ")[0][1:].lower()
            error_message += (
                f"<<ARGUMENT{variable_index}>> is a Variable returned by {api_name}, "
                f"it can only be used as final answer."
            )
        if error_message is not None:
            error_message += (
                f" Remember, Variables (<<CALLABLE_VARIABLE_LIST_STR>>) returned by get_relations, get_neighbors, intersection, get_attributes "
                f"can be used as inputs for subsequent actions or as final answers, "
                f"and Variables (<<NOT_CALLABLE_VARIABLE_LIST_STR>>) returned by get_relations, get_neighbors, intersection, get_attributes "
                f"can only be used as final answers."
            )
            raise KnowledgeGraphAPIException(error_message)

    def reset_cache(self) -> None:
        self.variable_to_relations_cache = {}
        self.variable_to_attributes_cache = {}

    @staticmethod
    def _construct_execution_message(observation: str) -> str:
        return f"<<API_STR>> executes successfully. Observation: {observation}"

    @staticmethod
    def _is_valid_entity(argument: str) -> bool:
        # According to https://www.wikidata.org/wiki/Property:P646
        # Freebase ID (or entity ID) can only start with 'g' or 'm', instead of 'm' or 'f'.
        if re.match(r"^([gm])\.[\w_]+$", argument):
            return True
        return False

    @staticmethod
    def _normalize_entity_id(text: str) -> Optional[str]:
        if text.startswith("http://rdf.freebase.com/ns/"):
            return text.replace("http://rdf.freebase.com/ns/", "")
        if KnowledgeGraphAPI._is_valid_entity(text):
            return text
        return None

    def _resolve_entity_ids(self, text: str, limit: int = 5) -> list[str]:
        cleaned = text.strip().strip('"').strip("'")
        normalized = self._normalize_entity_id(cleaned)
        if normalized:
            return [normalized]

        candidates = [cleaned]
        if cleaned.endswith("s") and len(cleaned) > 3:
            candidates.append(cleaned[:-1])

        for candidate in candidates:
            matches = self.sparql_executor.find_entities_by_name(candidate, limit=limit)
            if matches:
                return matches
        return []

    @staticmethod
    def _filter_noise_relations(relations: list[str]) -> list[str]:
        noise = {
            "type.object.name",
            "type.object.type",
            "type.object.key",
            "type.object.id",
            "common.topic.alias",
            "common.topic.description",
        }
        return [rel for rel in relations if rel not in noise]

    def final_execute(self, variable: Variable) -> list[str]:
        program = variable.program
        processed_code = LogicFormUtil.postprocess_raw_code(program)
        sparql_query = LogicFormUtil.lisp_to_sparql(processed_code)
        results = self.sparql_executor.execute_query(sparql_query)
        return results

    def get_relations(self, argument: Union[Variable, str]) -> tuple[None, str]:
        logger = logging.getLogger(__name__)
        resolved_candidates: list[str] = []
        debug_override = False
        # region Validate argument
        if isinstance(argument, Variable):
            KnowledgeGraphAPI._validate_variable("get_relations", [argument])
        else:
            if argument == "__DEBUG_GOAT__":
                resolved_candidates = ["m.03fwl"]
                debug_override = True
            else:
                resolved_candidates = self._resolve_entity_ids(argument)
            if not resolved_candidates:
                raise KnowledgeGraphAPIException(
                    "get_relations: <<ARGUMENT0>> is neither a Variable nor an entity. "
                    "The argument of get_relations must be a Variable or an entity."
                )
        # endregion
        if isinstance(argument, Variable):
            program = argument.program
            processed_code = LogicFormUtil.postprocess_raw_code(program)
            sparql_query = LogicFormUtil.lisp_to_sparql(processed_code)
            clauses = sparql_query.split("\n")
            new_clauses = [
                clauses[0],
                "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n{",
            ]
            new_clauses.extend(clauses[1:])
            new_clauses.append("}\n}")
            new_query = "\n".join(new_clauses)
            out_relations = self.sparql_executor.execute_query(new_query)
            resolved_entity = None
            raw_predicates: list[str] = []
        else:
            resolved_entity = resolved_candidates[0]
            out_relations = self.sparql_executor.get_out_relations(resolved_entity)
            raw_predicates = []
            try:
                raw_query = (
                    "SELECT ?p WHERE { "
                    f"<http://rdf.freebase.com/ns/{resolved_entity}> ?p ?o . "
                    "} LIMIT 2000"
                )
                raw_results = self.sparql_executor.execute_raw(raw_query)
                for result in raw_results["results"]["bindings"]:
                    raw_predicates.append(result["p"]["value"])
            except Exception:
                raw_predicates = []

        raw_predicate_count = len(raw_predicates)
        raw_relations = sorted(set(out_relations))
        if self.relations:
            intersected = sorted(
                list(set(raw_relations).intersection(set(self.relations)))
            )
            filtered_relations = intersected
        else:
            filtered_relations = self._filter_noise_relations(raw_relations)

        fail_open = False
        if not filtered_relations and raw_predicates:
            raw_predicates_clean = [
                rel.replace("http://rdf.freebase.com/ns/", "")
                for rel in raw_predicates
            ]
            filtered_relations = self._filter_noise_relations(
                sorted(set(raw_predicates_clean))
            )
            fail_open = True

        raw_predicate_sample = raw_predicates[:10]
        logger.info(
            "KG get_relations: input=%s resolved_candidates=%s chosen=%s debug=%s endpoint=%s raw_count=%d raw_predicate_count=%d filtered_count=%d fail_open=%s raw_sample=%s raw_pred_sample=%s",
            argument,
            resolved_candidates,
            resolved_entity,
            debug_override,
            self.sparql_executor.get_endpoint_url(),
            len(raw_relations),
            raw_predicate_count,
            len(filtered_relations),
            fail_open,
            raw_relations[:10],
            raw_predicate_sample,
        )
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"[{', '.join(filtered_relations)}]"
        )
        self.variable_to_relations_cache[argument] = filtered_relations
        return None, execution_message

    def get_neighbors(
        self, argument: Union[Variable, str], relation: str
    ) -> tuple[Variable, str]:
        logger = logging.getLogger(__name__)
        # region Validate arguments
        if isinstance(argument, Variable):
            KnowledgeGraphAPI._validate_variable("get_neighbors", [argument])
        else:
            resolved = self._resolve_entity_ids(argument)
            if not resolved:
                raise KnowledgeGraphAPIException(
                    "get_neighbors: <<ARGUMENT0>> is neither a Variable nor an entity. "
                    "The first argument of get_neighbors must be a Variable or an entity."
                )
        if argument not in self.variable_to_relations_cache.keys():
            raise KnowledgeGraphAPIException(
                f"get_neighbors: Execute get_relations for <<ARGUMENT0>> before executing get_neighbors"
            )
        if relation not in self.variable_to_relations_cache[argument]:
            raise KnowledgeGraphAPIException(
                f"get_neighbors: <<ARGUMENT1>> is not a relation of the <<ARGUMENT0>>. "
                f"<<ARGUMENT0>> has the following relations: {self.variable_to_relations_cache[argument]}"
            )
        # endregion
        resolved_entity = resolved[0] if not isinstance(argument, Variable) else None
        logger.info(
            "KG get_neighbors: input=%s resolved=%s relation=%s endpoint=%s",
            argument,
            resolved_entity,
            relation,
            self.sparql_executor.get_endpoint_url(),
        )
        range_type = self.range_info.get(relation)
        if not range_type:
            logger.warning(
                "KG missing range type for relation=%s; ontology_dir=%s; range_info_size=%d",
                relation,
                self.ontology_dir,
                len(self.range_info),
            )
            range_type = "type.object"
        new_variable = Variable(
            type=range_type,
            program=f"(JOIN {relation + '_inv'} {argument.program if isinstance(argument, Variable) else resolved_entity})",
        )
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"Variable <<NEW_VARIABLE>>, which are instances of {range_type}"
        )
        return new_variable, execution_message

    @staticmethod
    def intersection(variable1: Variable, variable2: Variable) -> tuple[Variable, str]:
        # region Validate arguments
        caller_name = "intersection"
        KnowledgeGraphAPI._ensure_variable(caller_name, [variable1, variable2])
        KnowledgeGraphAPI._validate_variable(caller_name, [variable1, variable2])
        if variable1.type != variable2.type:
            raise KnowledgeGraphAPIException(
                "intersection: Two Variables must have the same type"
            )
        # endregion
        new_variable = Variable(
            type=variable1.type,
            program=f"(AND {variable1.program} {variable2.program})",
        )
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"Variable <<NEW_VARIABLE>>, which are instances of {variable1.type}"
        )
        return new_variable, execution_message

    @staticmethod
    def union(variable1: Variable, variable2: Variable) -> tuple[Variable, str]:
        # region Validate arguments
        # The function is not included in the prompt
        KnowledgeGraphAPI._ensure_variable("union", [variable1, variable2])

        if variable1.type != variable2.type:
            raise KnowledgeGraphAPIException(
                "union: Two Variables must have the same type"
            )
        # endregion
        new_variable = Variable(
            type=variable1.type, program=f"(OR {variable1.program} {variable2.program})"
        )
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"Variable <<NEW_VARIABLE>>, which are instances of {variable1.type}"
        )
        return new_variable, execution_message

    @staticmethod
    def count(variable: Variable) -> tuple[Variable, str]:
        # region Validate arguments
        caller_name = "count"
        KnowledgeGraphAPI._ensure_variable(caller_name, [variable])
        KnowledgeGraphAPI._validate_variable(caller_name, [variable])
        # endregion
        new_variable = Variable(type="type.int", program=f"(COUNT {variable.program})")
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"Variable <<NEW_VARIABLE>>, which is a number"
        )
        return new_variable, execution_message

    def get_attributes(self, variable: Variable) -> tuple[None, str]:
        # region Validate variable
        caller_name = "get_attributes"
        KnowledgeGraphAPI._ensure_variable(caller_name, [variable])
        KnowledgeGraphAPI._validate_variable(caller_name, [variable])
        # endregion
        program = variable.program
        processed_code = LogicFormUtil.postprocess_raw_code(program)
        sparql_query = LogicFormUtil.lisp_to_sparql(processed_code)
        clauses = sparql_query.split("\n")
        new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n{"]
        new_clauses.extend(clauses[1:])
        new_clauses.append("}\n}")
        new_query = "\n".join(new_clauses)
        out_relations = self.sparql_executor.execute_query(new_query)
        out_relations = sorted(
            list(set(out_relations).intersection(set(self.attributes)))
        )
        self.variable_to_attributes_cache[variable] = out_relations
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"[{', '.join(out_relations)}]"
        )
        return None, execution_message

    def _find_extremum_by_attribute(
        self,
        variable: Variable,
        attribute: str,
        extremum_function: "KnowledgeGraphAPI.ExtremumFunction",
    ) -> tuple[Variable, str]:
        # region Validate arguments
        caller_name = inspect.stack()[1].function
        KnowledgeGraphAPI._ensure_variable(caller_name, [variable])
        KnowledgeGraphAPI._validate_variable(caller_name, [variable])
        self._validate_attribute(caller_name, variable, attribute)
        # endregion
        match extremum_function:
            case KnowledgeGraphAPI.ExtremumFunction.ARGMAX:
                function_name = "ARGMAX"
            case KnowledgeGraphAPI.ExtremumFunction.ARGMIN:
                function_name = "ARGMIN"
            case _:
                raise ValueError("This cannot happen.")
        new_variable = Variable(
            type=variable.type,
            program=f"({function_name} {variable.program} {attribute})",
        )
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"Variable <<NEW_VARIABLE>>, which are instances of {variable.type}"
        )
        return new_variable, execution_message

    def argmax(self, variable: Variable, attribute: str) -> tuple[Variable, str]:
        return self._find_extremum_by_attribute(
            variable, attribute, KnowledgeGraphAPI.ExtremumFunction.ARGMAX
        )

    def argmin(self, variable: Variable, attribute: str) -> tuple[Variable, str]:
        return self._find_extremum_by_attribute(
            variable, attribute, KnowledgeGraphAPI.ExtremumFunction.ARGMIN
        )

    @staticmethod
    def get_valid_api_name_list() -> list[str]:
        return [
            # "final_execute",
            "get_relations",
            "get_neighbors",
            "intersection",
            # "union",
            "count",
            "get_attributes",
            "argmax",
            "argmin",
        ]

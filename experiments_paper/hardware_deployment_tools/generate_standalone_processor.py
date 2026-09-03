#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Generate an EnvPool-free processor header and an in-place pybind wrapper.

The generator targets processors that directly implement the
``GazeboProcessor<EnvSpec>`` interface.  It deliberately fails on source
constructs it cannot translate safely instead of emitting subtly different
deployment code. Quoted headers directly included by the processor are copied
into the destination root and their include paths are flattened. Includes of
those copied headers are intentionally not traversed.

Example
-------
python experiments_paper/hardware_deployment_tools/generate_standalone_processor.py \
    envs/processors/trajectory_tracking_processor.hh \
    /tmp/trajectory_tracking_processor \
    --class-name TrajectoryTrackingProcessor \
    --action-dimension 3 \
    --module-name trajectory_tracking_processor
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import gzdrl


class GenerationError(RuntimeError):
    """Raised when a processor cannot be transformed without guessing."""


@dataclass(frozen=True)
class ConstructorArgument:
    cpp_type: str
    name: str


@dataclass(frozen=True)
class DirectInclude:
    include: str
    source: Path
    output_name: str


@dataclass(frozen=True)
class GenerationResult:
    header: str
    binding: str
    class_name: str
    observation_allocation_added: bool
    direct_includes: tuple[DirectInclude, ...]


def _identifier(value: str, description: str) -> str:
    if not re.fullmatch(r"[A-Za-z_]\w*", value):
        raise GenerationError(
            f"{description} must be a valid C++ identifier: {value!r}"
        )
    return value


def _class_candidates(source: str) -> list[str]:
    return re.findall(
        r"class\s+([A-Za-z_]\w*)\s*"
        r":\s*public\s+GazeboProcessor\s*<",
        source,
    )


def _select_class(source: str, requested: str | None) -> str:
    candidates = _class_candidates(source)
    if requested is not None:
        _identifier(requested, "class name")
        if requested not in candidates:
            raise GenerationError(
                f"{requested!r} is not a direct GazeboProcessor subclass. "
                f"Detected candidates: {candidates or 'none'}"
            )
        return requested
    if len(candidates) != 1:
        raise GenerationError(
            "Could not select one processor class automatically. Pass "
            f"--class-name. Detected candidates: {candidates or 'none'}"
        )
    return candidates[0]


def _remove_envpool_includes(source: str) -> str:
    kept: list[str] = []
    for line in source.splitlines(keepends=True):
        match = re.match(r"\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", line)
        if match:
            include = match.group(1).lower()
            if (
                "envpool" in include
                or "gazebo" in include
                or include == "processors/processor.hh"
            ):
                continue
        kept.append(line)
    return "".join(kept)


def _automatic_include_roots(input_path: Path) -> list[Path]:
    roots = [
        input_path.parent,
        input_path.parent.parent,
        gzdrl.get_include_path(),
    ]
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def _flatten_direct_includes(
    source: str,
    input_path: Path,
    explicit_include_roots: Sequence[Path],
) -> tuple[str, tuple[DirectInclude, ...]]:
    roots: list[Path] = []
    seen_roots: set[Path] = set()
    for root in [
        *(path.expanduser().resolve() for path in explicit_include_roots),
        *_automatic_include_roots(input_path),
    ]:
        if root not in seen_roots:
            roots.append(root)
            seen_roots.add(root)

    include_pattern = re.compile(
        r'(?m)^(?P<prefix>[ \t]*#[ \t]*include[ \t]*)'
        r'"(?P<include>[^"\n]+)"'
    )
    dependencies: dict[str, DirectInclude] = {}

    def replace_include(match: re.Match[str]) -> str:
        include = match.group("include")
        resolved_source = next(
            (
                candidate
                for root in roots
                if (candidate := (root / include).resolve()).is_file()
            ),
            None,
        )
        if resolved_source is None:
            searched = ", ".join(str(root) for root in roots)
            raise GenerationError(
                f"Could not resolve direct include {include!r}. Searched: "
                f"{searched}. Add its parent with --include-root."
            )

        output_name = resolved_source.name
        dependency = DirectInclude(
            include=include,
            source=resolved_source,
            output_name=output_name,
        )
        existing = dependencies.get(output_name)
        if existing is not None and existing.source != dependency.source:
            raise GenerationError(
                "Direct includes cannot be flattened because two different "
                f"files have the basename {output_name!r}: "
                f"{existing.source} and {dependency.source}."
            )
        dependencies[output_name] = dependency
        return f'{match.group("prefix")}"{output_name}"'

    transformed = include_pattern.sub(replace_include, source)
    return transformed, tuple(dependencies.values())


def _concretize_processor(
    source: str,
    class_name: str,
    action_dimension: int | None,
    state_dimension: int,
) -> tuple[str, int]:
    pattern = re.compile(
        r"template\s*<(?P<parameters>[^>]*)>\s*"
        r"class\s+"
        + re.escape(class_name)
        + r"\s*:\s*public\s+GazeboProcessor\s*<\s*"
        r"(?P<env_spec>[A-Za-z_]\w*)\s*>"
    )
    match = pattern.search(source)
    if match is None:
        raise GenerationError(
            f"Could not locate the templated declaration for {class_name}. "
            "Only direct GazeboProcessor subclasses are supported."
        )

    parameters = match.group("parameters")
    env_spec = match.group("env_spec")
    action_parameter_match = re.search(
        r"\bint\s+([A-Za-z_]\w*)\s*=\s*(\d+)", parameters
    )
    if action_dimension is None:
        concrete_dimension = re.search(
            r"static\s+constexpr\s+int\s+kActionDimension\s*=\s*(\d+)\s*;",
            source,
        )
        if action_parameter_match is not None:
            action_dimension = int(action_parameter_match.group(2))
        elif concrete_dimension is not None:
            action_dimension = int(concrete_dimension.group(1))
        else:
            raise GenerationError(
                "The action dimension is neither a defaulted integer template "
                "parameter nor a concrete kActionDimension. Pass "
                "--action-dimension explicitly."
            )
    if action_dimension <= 0:
        raise GenerationError("The action dimension must be positive.")

    transformed = source[: match.start()] + f"class {class_name}" + source[match.end() :]

    transformed, state_count = re.subn(
        r"\s*using\s+State\s*=\s*typename\s+GazeboProcessor\s*<\s*"
        + re.escape(env_spec)
        + r"\s*>\s*::\s*State\s*;",
        "",
        transformed,
        count=1,
    )
    transformed, action_count = re.subn(
        r"\s*using\s+Action\s*=\s*typename\s+GazeboProcessor\s*<\s*"
        + re.escape(env_spec)
        + r"\s*>\s*::\s*Action\s*;",
        "",
        transformed,
        count=1,
    )
    aliases = (
        "\n    using Statef = Eigen::Matrix<float, "
        f"{state_dimension}, 1>;\n"
        "    using StateMap = std::unordered_map<std::string, Statef>;\n"
        "    using VectorMap = "
        "std::unordered_map<std::string, Eigen::VectorXf>;\n"
        "    using State = VectorMap;\n"
        "    using Action = VectorMap;"
    )
    transformed, statef_count = re.subn(
        r"\s*using\s+Statef\s*=\s*typename\s+GazeboProcessor\s*<\s*"
        + re.escape(env_spec)
        + r"\s*>\s*::\s*Statef\s*;",
        aliases,
        transformed,
        count=1,
    )
    if (state_count, action_count, statef_count) != (1, 1, 1):
        raise GenerationError(
            "Could not replace the GazeboProcessor State, Action, and Statef "
            "aliases. The processor interface differs from the supported form."
        )

    if action_parameter_match is not None:
        action_parameter = action_parameter_match.group(1)
        transformed, count = re.subn(
            r"(static\s+constexpr\s+int\s+kActionDimension\s*=\s*)"
            + re.escape(action_parameter)
            + r"(\s*;)",
            rf"\g<1>{action_dimension}\g<2>",
            transformed,
            count=1,
        )
        if count != 1:
            raise GenerationError(
                "Could not concretize kActionDimension from the action "
                "template parameter."
            )
        if re.search(r"\b" + re.escape(action_parameter) + r"\b", transformed):
            raise GenerationError(
                f"Template parameter {action_parameter} is still used after "
                "concretization."
            )
    else:
        transformed, dimension_count = re.subn(
            r"(static\s+constexpr\s+int\s+kActionDimension\s*=\s*)"
            r"\d+(\s*;)",
            rf"\g<1>{action_dimension}\g<2>",
            transformed,
            count=1,
        )
        if dimension_count == 0:
            transformed = transformed.replace(
                "    using Action = VectorMap;",
                "    using Action = VectorMap;\n\n"
                "    static constexpr int kActionDimension = "
                f"{action_dimension};",
                1,
            )

    if re.search(r"\b" + re.escape(env_spec) + r"\b", transformed):
        raise GenerationError(
            f"EnvSpec template parameter {env_spec} is still used after "
            "concretization."
        )
    return transformed, action_dimension


def _replace_envpool_keys(source: str) -> str:
    # EnvPool's compile-time key literal: "obs"_ -> ordinary map key "obs".
    return re.sub(r'"((?:[^"\\]|\\.)*)"_', r'"\1"', source)


def _replace_action_data_access(source: str) -> str:
    pattern = re.compile(
        r"const\s+float\s*\*\s*action_data\s*=\s*"
        r"static_cast\s*<\s*const\s+float\s*\*\s*>\s*"
        r"\(\s*policy_action\s*\[\s*\"action\"\s*\]\.Data\s*\(\s*\)\s*\)\s*;"
    )
    replacement = """const Eigen::VectorXf &action_data =
            policy_action.at("action");
        if (action_data.size() != kActionDimension)
        {
            throw std::invalid_argument(
                "Policy action has dimension " +
                std::to_string(action_data.size()) + ", expected " +
                std::to_string(kActionDimension) + ".");
        }"""
    transformed, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise GenerationError(
            "Could not replace the EnvPool policy_action Data() access."
        )
    if ".Data()" in transformed or re.search(r'"[^"\n]+"_', transformed):
        raise GenerationError(
            "EnvPool array/key operations remain after transformation."
        )
    return transformed


def _add_observation_allocation(source: str) -> tuple[str, bool]:
    keys = sorted(
        set(
            re.findall(
                r'processed_obs\s*\[\s*"([^"\n]+)"\s*\]\s*\[\s*index\+\+\s*\]',
                source,
            )
        )
    )
    if len(keys) != 1 or "ExpectedObservationDimension()" not in source:
        return source, False

    key = keys[0]
    pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)int\s+index\s*=\s*0\s*;"
    )
    match = pattern.search(source)
    if match is None:
        return source, False
    indent = match.group("indent")
    allocation = (
        f'{indent}auto &observation = processed_obs["{key}"];\n'
        f"{indent}observation.resize(ExpectedObservationDimension());\n"
        f"{indent}int index = 0;"
    )
    transformed = source[: match.start()] + allocation + source[match.end() :]
    transformed = re.sub(
        r'processed_obs\s*\[\s*"'
        + re.escape(key)
        + r'"\s*\]\s*\[\s*index\+\+\s*\]',
        "observation[index++]",
        transformed,
    )

    # Expose the computed size to C++ and Python callers without making the
    # existing implementation helper public.
    public_size = (
        "    int ObservationDimension() const noexcept\n"
        "    {\n"
        "        return ExpectedObservationDimension();\n"
        "    }\n\n"
    )
    transformed, count = re.subn(
        r"(?m)^(private:\s*)$", public_size + r"\1", transformed, count=1
    )
    if count != 1:
        raise GenerationError(
            "Could not expose the inferred observation dimension."
        )
    return transformed, True


def _standalone_preamble(source: str, input_path: Path) -> str:
    guard_match = re.search(
        r"#ifndef\s+([A-Za-z_]\w*)\s*\n#define\s+\1", source
    )
    if guard_match:
        old_guard = guard_match.group(1)
        new_guard = "STANDALONE_" + old_guard
        source = re.sub(r"\b" + re.escape(old_guard) + r"\b", new_guard, source)

    banner = (
        "// Generated by experiments_paper/hardware_deployment_tools/"
        "generate_standalone_processor.py from\n"
        f"// {input_path.as_posix()}. Do not edit this generated copy by hand.\n"
    )
    include = "#include <unordered_map>\n"
    first_include = source.find("#include")
    if first_include < 0:
        raise GenerationError("The processor header has no include block.")
    if re.search(r"#include\s*<unordered_map>", source) is None:
        source = source[:first_include] + include + source[first_include:]
    return banner + source


def _find_matching_parenthesis(source: str, opening: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(opening, len(source)):
        character = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in ('"', "'"):
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    raise GenerationError("Unbalanced constructor parameter list.")


def _split_top_level(value: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depths = {"<": 0, "(": 0, "[": 0, "{": 0}
    closing = {">": "<", ")": "(", "]": "[", "}": "{"}
    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in ('"', "'"):
            quote = character
        elif character in depths:
            depths[character] += 1
        elif character in closing:
            opener = closing[character]
            depths[opener] = max(0, depths[opener] - 1)
        elif character == separator and all(depth == 0 for depth in depths.values()):
            parts.append(value[start:index].strip())
            start = index + 1
    final = value[start:].strip()
    if final:
        parts.append(final)
    return parts


def _remove_top_level_default(argument: str) -> str:
    depths = {"<": 0, "(": 0, "[": 0, "{": 0}
    closing = {">": "<", ")": "(", "]": "[", "}": "{"}
    for index, character in enumerate(argument):
        if character in depths:
            depths[character] += 1
        elif character in closing:
            opener = closing[character]
            depths[opener] = max(0, depths[opener] - 1)
        elif character == "=" and all(depth == 0 for depth in depths.values()):
            return argument[:index].strip()
    return argument.strip()


def _constructor_arguments(source: str, class_name: str) -> list[ConstructorArgument]:
    match = re.search(
        r"(?m)^\s*" + re.escape(class_name) + r"\s*\(", source
    )
    if match is None:
        raise GenerationError(f"Could not find the {class_name} constructor.")
    opening = source.find("(", match.start())
    closing = _find_matching_parenthesis(source, opening)
    parameters = source[opening + 1 : closing]
    result: list[ConstructorArgument] = []
    for raw_argument in _split_top_level(parameters):
        argument = _remove_top_level_default(raw_argument)
        name_match = re.search(r"([A-Za-z_]\w*)\s*$", argument)
        if name_match is None:
            raise GenerationError(
                f"Could not parse constructor argument: {raw_argument!r}"
            )
        name = name_match.group(1)
        cpp_type = argument[: name_match.start()].strip()
        if not cpp_type:
            raise GenerationError(
                f"Constructor argument {name!r} has no detected C++ type."
            )
        result.append(ConstructorArgument(cpp_type=cpp_type, name=name))
    if not result:
        raise GenerationError("A default-only constructor is not supported.")
    return result


def _indent_lines(values: Iterable[str], spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + value for value in values)


def _binding_source(
    header_name: str,
    class_name: str,
    module_name: str,
    constructor: Sequence[ConstructorArgument],
    processor_source: str,
    observation_allocation_added: bool,
) -> str:
    init_types = ",\n".join(argument.cpp_type for argument in constructor)
    py_args = ",\n".join(
        f'py::arg("{argument.name}")' for argument in constructor
    )

    method_bindings: list[str] = []
    if re.search(r"\bvoid\s+UpdateTrajectory\s*\(", processor_source):
        method_bindings.append(
            f'.def("update_trajectory", &{class_name}::UpdateTrajectory)'
        )
    if re.search(r"\bvoid\s+Reset\s*\(", processor_source):
        method_bindings.append(
            f'.def("reset", &{class_name}::Reset, py::arg("initial_state"))'
        )

    method_bindings.append(
        """.def(
            "process_observation",
            [](Processor &self,
               const Processor::StateMap &current_state,
               const Processor::StateMap &current_state_dot,
               const Processor::StateMap &previous_state,
               const Processor::StateMap &previous_state_dot,
               py::dict processed_observation)
            {
                Processor::VectorMap output =
                    VectorMapFromDict(processed_observation);
                {
                    py::gil_scoped_release release;
                    self.ProcessObservation(
                        current_state, current_state_dot, previous_state,
                        previous_state_dot, output);
                }
                UpdateVectorDict(processed_observation, output);
            },
            py::arg("current_state"), py::arg("current_state_dot"),
            py::arg("previous_state"), py::arg("previous_state_dot"),
            py::arg("processed_observation"))"""
    )
    method_bindings.append(
        """.def(
            "process_action",
            [](Processor &self, py::dict policy_action,
               py::dict processed_action)
            {
                const Processor::VectorMap input =
                    VectorMapFromDict(policy_action);
                Processor::VectorMap output =
                    VectorMapFromDict(processed_action);
                {
                    py::gil_scoped_release release;
                    self.ProcessAction(input, output);
                }
                UpdateVectorDict(processed_action, output);
            },
            py::arg("policy_action"), py::arg("processed_action"))"""
    )
    if observation_allocation_added:
        method_bindings.append(
            f'.def_property_readonly("observation_dimension", '
            f'&{class_name}::ObservationDimension)'
        )
    if re.search(r"\bStatef\s+ReferenceState\s*\(", processor_source):
        method_bindings.append(
            f'.def("reference_state", &{class_name}::ReferenceState)'
        )
    if re.search(r"\bStatef\s+ReferenceStateAt\s*\(", processor_source):
        method_bindings.append(
            f'.def("reference_state_at", &{class_name}::ReferenceStateAt, '
            'py::arg("time"))'
        )
    if re.search(r"\bfloat\s+CurrentTime\s*\(", processor_source):
        method_bindings.append(
            f'.def_property_readonly("current_time", &{class_name}::CurrentTime)'
        )

    chain = "\n        ".join(method_bindings)
    return f'''// Generated by experiments_paper/hardware_deployment_tools/generate_standalone_processor.py.
#include "{header_name}"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using Processor = {class_name};

namespace
{{
Processor::VectorMap VectorMapFromDict(const py::dict &source)
{{
    Processor::VectorMap result;
    for (const auto &item : source)
    {{
        result.emplace(
            py::cast<std::string>(item.first),
            py::cast<Eigen::VectorXf>(item.second));
    }}
    return result;
}}

void UpdateVectorDict(
    py::dict target, const Processor::VectorMap &source)
{{
    for (const auto &[key, vector] : source)
    {{
        const py::str python_key(key);
        bool updated_existing_array = false;
        if (target.contains(python_key))
        {{
            const py::handle existing = target[python_key];
            if (py::isinstance<py::array>(existing))
            {{
                py::array array = py::reinterpret_borrow<py::array>(existing);
                const bool compatible =
                    array.dtype().is(py::dtype::of<float>()) &&
                    array.ndim() == 1 &&
                    array.shape(0) == vector.size() &&
                    (array.flags() & py::array::c_style) != 0 &&
                    array.writeable();
                if (compatible)
                {{
                    std::copy(
                        vector.data(), vector.data() + vector.size(),
                        static_cast<float *>(array.mutable_data()));
                    updated_existing_array = true;
                }}
            }}
        }}
        if (!updated_existing_array)
        {{
            target[python_key] = py::cast(vector);
        }}
    }}
}}
}} // namespace

PYBIND11_MODULE({module_name}, module)
{{
    module.doc() =
        "Standalone processor with in-place Python output dictionaries.";

    py::class_<Processor>(module, "{class_name}")
        .def(
            py::init<
{_indent_lines(init_types.splitlines(), 16)}
            >(),
{_indent_lines(py_args.splitlines(), 12)})
        {chain};
}}
'''


def generate(
    input_path: Path,
    class_name: str | None,
    action_dimension: int | None,
    state_dimension: int,
    module_name: str,
    include_roots: Sequence[Path] = (),
) -> GenerationResult:
    source = input_path.read_text(encoding="utf-8")
    selected_class = _select_class(source, class_name)
    transformed = _remove_envpool_includes(source)
    transformed, direct_includes = _flatten_direct_includes(
        transformed, input_path, include_roots
    )
    transformed, _ = _concretize_processor(
        transformed,
        selected_class,
        action_dimension,
        state_dimension,
    )
    transformed = _replace_envpool_keys(transformed)
    transformed = _replace_action_data_access(transformed)
    transformed = re.sub(r"\s+override(?=\s*(?:\{|;))", "", transformed)
    transformed, allocation_added = _add_observation_allocation(transformed)
    transformed = _standalone_preamble(transformed, input_path)

    forbidden = {
        "GazeboProcessor": "GazeboProcessor inheritance/type",
        "envpool/": "EnvPool include",
    }
    for token, description in forbidden.items():
        if token in transformed:
            raise GenerationError(
                f"Generated header still contains {description}: {token!r}"
            )

    constructor = _constructor_arguments(transformed, selected_class)
    header_name = input_path.name
    binding = _binding_source(
        header_name,
        selected_class,
        module_name,
        constructor,
        transformed,
        allocation_added,
    )
    return GenerationResult(
        header=transformed,
        binding=binding,
        class_name=selected_class,
        observation_allocation_added=allocation_added,
        direct_includes=direct_includes,
    )


def _write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise GenerationError(
            f"Refusing to overwrite {path}. Pass --force to replace it."
        )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _copy_file(source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        raise GenerationError(
            f"Refusing to overwrite {destination}. Pass --force to replace it."
        )
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a standalone processor header and pybind11 binding from "
            "a direct GazeboProcessor subclass."
        )
    )
    parser.add_argument("processor", type=Path, help="Input processor header")
    parser.add_argument("destination", type=Path, help="Output directory")
    parser.add_argument(
        "--class-name",
        help="Processor class; auto-detected when the input has exactly one",
    )
    parser.add_argument(
        "--action-dimension",
        type=int,
        default=None,
        help=(
            "Concrete action dimension. Defaults to a template default or "
            "the processor's concrete kActionDimension."
        ),
    )
    parser.add_argument(
        "--state-dimension",
        type=int,
        default=13,
        help="Raw state vector dimension (default: 13)",
    )
    parser.add_argument(
        "--module-name",
        default="standalone_processor",
        help="Generated Python extension module name",
    )
    parser.add_argument(
        "--include-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional root used to resolve quoted processor includes; may "
            "be repeated. Direct includes are copied, but their own includes "
            "are not traversed."
        ),
    )
    parser.add_argument(
        "--binding-name",
        default="binding.cc",
        help="Generated binding source filename",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated files in the destination",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        processor = args.processor.expanduser().resolve()
        if not processor.is_file():
            raise GenerationError(f"Processor header does not exist: {processor}")
        if args.state_dimension <= 0:
            raise GenerationError("The state dimension must be positive.")
        module_name = _identifier(args.module_name, "module name")
        binding_name = Path(args.binding_name)
        if binding_name.name != args.binding_name or binding_name.suffix != ".cc":
            raise GenerationError(
                "--binding-name must be a plain filename ending in .cc."
            )

        destination = args.destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        header_path = destination / processor.name
        binding_path = destination / binding_name
        if header_path == processor:
            raise GenerationError(
                "Destination would overwrite the source processor. Choose a "
                "different directory."
            )

        result = generate(
            processor,
            args.class_name,
            args.action_dimension,
            args.state_dimension,
            module_name,
            args.include_root,
        )
        dependency_paths = tuple(
            destination / dependency.output_name
            for dependency in result.direct_includes
        )
        output_paths = (header_path, binding_path, *dependency_paths)
        if len(set(output_paths)) != len(output_paths):
            raise GenerationError(
                "A copied direct include conflicts with a generated output "
                "filename. Rename the binding or use a different processor."
            )
        existing_outputs = [path for path in output_paths if path.exists()]
        if existing_outputs and not args.force:
            formatted = ", ".join(str(path) for path in existing_outputs)
            raise GenerationError(
                "Refusing to overwrite existing generated file(s): "
                f"{formatted}. Pass --force to replace them."
            )

        _write(header_path, result.header, args.force)
        _write(binding_path, result.binding, args.force)
        for dependency, output_path in zip(
            result.direct_includes, dependency_paths
        ):
            _copy_file(dependency.source, output_path, args.force)

        print(f"Generated standalone header: {header_path}")
        print(f"Generated pybind source:     {binding_path}")
        for dependency, output_path in zip(
            result.direct_includes, dependency_paths
        ):
            print(
                f"Copied direct include:        {output_path} "
                f"(from {dependency.include})"
            )
        if not result.observation_allocation_added:
            print(
                "Warning: observation output size could not be inferred; "
                "callers must preallocate the processed-observation vectors.",
                file=sys.stderr,
            )
        return 0
    except (GenerationError, OSError, UnicodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

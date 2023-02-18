from __future__ import annotations

import ast
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from expression import (
    Alias,
    ApplyFuncArg,
    ApplyFuncArgColumn,
    BinaryBooleanExpr,
    BinaryOp,
    Conditional,
    Constant,
    Expr,
    FString,
    NegateBooleanExpr,
    StrMethodCall,
    TrivialFuncCall,
    TypeCastFuncCall,
    convert_expr_to_vectorized_code,
)


def get_op_symbol(operator: ast.operator) -> str:
    if isinstance(operator, ast.Add):
        return "+"
    if isinstance(operator, ast.Sub):
        return "-"
    if isinstance(operator, ast.Mult):
        return "*"
    if isinstance(operator, ast.Div):
        return "/"
    if isinstance(operator, ast.Pow):
        return "**"
    raise NotImplementedError(type(operator))


def get_comp_symbol(operator: ast.cmpop) -> str:
    if isinstance(operator, ast.Eq):
        return "=="
    if isinstance(operator, ast.NotEq):
        return "!="
    if isinstance(operator, ast.Gt):
        return ">"
    if isinstance(operator, ast.GtE):
        return ">="
    if isinstance(operator, ast.Lt):
        return "<"
    if isinstance(operator, ast.LtE):
        return "<="
    raise NotImplementedError(type(operator))


# Default packages alias if the input code omits the import line.
DEFAULT_PANDAS_ALIAS = "pd"
DEFAULT_NUMPY_ALIAS = "np"

REPLACEABLE_PANDAS_FUNCS = {"isna", "isnull"}
REPLACEABLE_NUMPY_FUNCS = {
    "isnan",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "hypot",
    "arctan2",
    "degrees",
    "radians",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arctanh",
    "around",
    "rint",
    "fix",
    "floor",
    "ceil",
    "trunc",
    "exp",
    "expm1",
    "exp2",
    "log",
    "log10",
    "log2",
    "log1p",
    "abs",
    "absolute",
    "square",
    "sqrt",
    "clip",
    "sign",
}

NATIVE_TYPE_FUNCS = ["int", "str", "float", "bool"]
NUMPY_TYPE_FUNCS = [
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float",
    "float16",
    "float32",
    "float64",
    "bool",
    "str",
]

STR_METHODS = [
    "capitalize",
    "casefold",
    "center",
    "count",
    "decode",
    "encode",
    "endswith",
    "find",
    "index",
    "isalnum",
    "isalpha",
    "isdecimal",
    "isdigit",
    "islower",
    "isnumeric",
    "isspace",
    "istitle",
    "isupper",
    "join",
    "len",
    "ljust",
    "lower",
    "lstrip",
    "partition",
    "replace",
    "rfind",
    "rindex",
    "rjust",
    "rpartition",
    "rsplit",
    "rstrip",
    "slice",
    "slice_replace",
    "split",
    "startswith",
    "strip",
    "swapcase",
    "title",
    "translate",
    "upper",
    "zfill",
]


@dataclass
class ApplyInfo:
    """
    Hold some information related to an assignment which right hand side is a call to `apply`.
    eg: s = s.apply(func)
    """

    assigned_var_name: str
    calling_var_name: str
    start_lineno: int
    end_lineno: int
    ast_func_expr: ast.expr  # The node corresponding to the func parameter passed to `apply`


@dataclass
class ResolvedAssign:
    var_name: str
    expr: Expr


class FunctionBodyParser:

    udf_name_to_func_def_node: Dict[str, ast.FunctionDef]
    input_code: str  # used for ast.get_source_segment

    pandas_alias: str
    numpy_alias: str

    def __init__(self, input_code: str):
        self.udf_name_to_func_def_node = {}
        self.input_code = input_code
        self.pandas_alias = DEFAULT_PANDAS_ALIAS
        self.numpy_alias = DEFAULT_NUMPY_ALIAS

    def _is_external_package_func(self, alias: Optional[str], func_name: str) -> bool:
        if alias is None:
            return False

        return (alias == self.pandas_alias and func_name in REPLACEABLE_PANDAS_FUNCS) or (
            alias == self.numpy_alias and func_name in REPLACEABLE_NUMPY_FUNCS
        )

    def _resolve_binary_op(self, binary_op_node: ast.BinOp, dependencies: Dict[str, Expr]) -> BinaryOp:
        # TODO Handle the fact that if left and/or right are Conditional, we should build a Conditional binary op.

        left = self._resolve_expr(binary_op_node.left, dependencies)
        right = self._resolve_expr(binary_op_node.right, dependencies)
        return BinaryOp(left=left, right=right, pd_symbol=get_op_symbol(binary_op_node.op))

    def _resolve_compare(self, compare_node: ast.Compare, dependencies: Dict[str, Expr]) -> BinaryBooleanExpr:
        assert len(compare_node.comparators) == 1, "multiple comparators not supported"
        assert len(compare_node.ops) == 1, "multiple comparison ops not supported"

        left = self._resolve_expr(compare_node.left, dependencies)
        right = self._resolve_expr(compare_node.comparators[0], dependencies)
        return BinaryBooleanExpr(
            left=left,
            right=right,
            pd_symbol=get_comp_symbol(compare_node.ops[0]),
        )

    def _resolve_subscript(self, subscript_node: ast.Subscript, dependencies: Dict[str, Expr]) -> ApplyFuncArgColumn:
        attr = subscript_node.value
        if isinstance(attr, ast.Name) and isinstance(dependencies[attr.id], ApplyFuncArg):
            if isinstance(subscript_node.slice, ast.Constant):
                # Do not use subscript_node.slice.value as it removes quotes from string constant.
                return ApplyFuncArgColumn(ast.get_source_segment(self.input_code, subscript_node.slice), dot_notation=False)  # type: ignore[arg-type]
        raise NotImplementedError(f"{subscript_node=}")

    def _resolve_attribute(self, attribute_node: ast.Attribute, dependencies: Dict[str, Expr]) -> ApplyFuncArgColumn:
        attr = attribute_node.value
        if isinstance(attr, ast.Name) and isinstance(dependencies[attr.id], ApplyFuncArg):
            return ApplyFuncArgColumn(attribute_node.attr, dot_notation=True)
        raise NotImplementedError(f"{attribute_node=}")

    def _resolve_fstring(self, fstring_node: ast.JoinedStr, dependencies: Dict[str, Expr]) -> FString:
        for item in fstring_node.values:
            if isinstance(item, ast.FormattedValue) and (item.conversion != -1 or item.format_spec is not None):
                raise NotImplementedError("fstring `conversion` or `format_spec` is not supported")

        # ast.JoinedStr.values contains a list of either ast.FormattedValue or ast.Constant
        items = [item.value if isinstance(item, ast.FormattedValue) else item for item in fstring_node.values]
        resolved_items = [self._resolve_expr(item, dependencies) for item in items]
        return FString(items=resolved_items)

    def _resolve_call(self, call_node: ast.Call, dependencies: Dict[str, Expr]) -> Expr:
        """
        Multiple cases to handle:
        - Calling a UDF.
        - Calling a builtin type cast function (int, str, ...).
        - Calling an external type cast function (np.int8, ...)
        - Calling a function from an external package (pd.isna, np.log, ...).
        - Calling a UDF from an eternal module -> NOT SUPPORTED.
        """

        func = call_node.func
        args_expr = [self._resolve_expr(arg, dependencies) for arg in call_node.args]

        if isinstance(func, ast.Name):
            if func.id in NATIVE_TYPE_FUNCS and len(args_expr) == 1:
                return TypeCastFuncCall(alias=None, func_name=func.id, param=args_expr[0])

            if func.id in self.udf_name_to_func_def_node:
                return self._parse_function_def(self.udf_name_to_func_def_node[func.id], args_expr)

        if isinstance(func, ast.Attribute):
            caller_expr = self._resolve_expr(func.value, dependencies)
            func_name = func.attr

            if func_name in STR_METHODS:
                return StrMethodCall(
                    caller=caller_expr,
                    method_name=func_name,
                    params=args_expr,
                )

            if isinstance(caller_expr, Alias):
                alias = caller_expr.name
                if self._is_external_package_func(alias, func_name):
                    return TrivialFuncCall(
                        alias=alias,
                        func_name=func_name,
                        params=args_expr,
                    )

                if alias == self.numpy_alias and func_name in NUMPY_TYPE_FUNCS and len(args_expr) == 1:
                    return TypeCastFuncCall(alias=alias, func_name=func_name, param=args_expr[0])

        raise NotImplementedError(f"{call_node=}")

    def _resolve_expr(self, expr_node: Optional[ast.expr], dependencies: Dict[str, Expr]) -> Expr:
        """
        Build an Expr object from a ast.expr node, using the current dependencies.
        """

        assert expr_node is not None, "Trying to resolve an empty expression node"

        if isinstance(expr_node, ast.Constant):
            return Constant(expr_node.value)

        if isinstance(expr_node, ast.Name):
            if expr_node.id == self.numpy_alias or expr_node.id == self.pandas_alias:
                return Alias(expr_node.id)
            try:
                return dependencies[expr_node.id]
            except KeyError:
                # Could be an external module, a global variable (not supported as of writing)
                raise NotImplementedError(f"Missing dependency {expr_node.id=}")

        if isinstance(expr_node, ast.BinOp):
            return self._resolve_binary_op(expr_node, dependencies)

        if isinstance(expr_node, ast.Compare):
            return self._resolve_compare(expr_node, dependencies)

        if isinstance(expr_node, ast.Call):
            return self._resolve_call(expr_node, dependencies)

        if isinstance(expr_node, ast.Subscript):
            return self._resolve_subscript(expr_node, dependencies)

        if isinstance(expr_node, ast.Attribute):
            return self._resolve_attribute(expr_node, dependencies)

        if isinstance(expr_node, ast.JoinedStr):
            return self._resolve_fstring(expr_node, dependencies)

        raise NotImplementedError(f"{expr_node}=")

    def _parse_function_def(self, func_def_node: ast.FunctionDef, args_expr: List[Expr]) -> Expr:
        """
        Initiate the parsing of the function body, and return the Expr object associated to the __RETURN__ dependency.
        See details in `parse_function_body` docstring.
        """

        args = func_def_node.args.args
        assert len(args) == len(args_expr)
        dependencies: Dict[str, Expr] = dict(zip([arg.arg for arg in args], args_expr))
        dependencies = self._parse_statements(func_def_node.body, dependencies, is_function_body=True)
        return dependencies["__RETURN__"]

    def _parse_statements(
        self, body: List[ast.stmt], dependencies: Dict[str, Expr], is_function_body: bool = False
    ) -> Dict[str, Expr]:
        """
        Iterate over a list of statements and return ONLY the new dependencies mapping created during these statements.
        It is useful to return only the new dependencies, as if we are parsing the body of an IF statement,
        we want to be able to know which statements were created within the IF block in order to flag them
        as conditional. However, the parsing of the statements must be performed within the current dependencies
        context, thus we need to both update the current (copy of) dependencies, and the new dependencies.

        `dependencies` indicates a mapping between a python variable and an Expression object.
        Only resolved dependencies are added to the dependencies. For instance:

        ```
        a = 1  # dependencies: {"a" -> Constant(1)}
        b = a  # dependencies: {"a" -> Constant(1), "b" -> Constant(1)}
        a = 2  # dependencies: {"a" -> Constant(2), "b" -> Constant(1)}

        On the second statement, we directly resolved the dependency "b" -> "a" to "b" -> Constant(1),
        as if "a" is reassigned later the dependency of "b" would be broken (-> Constant(2)).
        ```

        When a return statement is met, we add a dependency to "__RETURN__". If a function has multiple return
        statements, these return are necessarily conditionals, and the "__RETURN__" expression should be updated
        accordingly using a Conditional object. Basically, when we meet a non-first return statement,
        the expression of this return goes into the field `else_` of the existing "__RETURN__" expression.
        Thus, we always keep a single "__RETURN__" dependency.
        """

        dependencies = deepcopy(dependencies)
        new_dependencies = {}
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                resolved_assign = self._parse_function_assign(stmt, dependencies)
                dependencies[resolved_assign.var_name] = resolved_assign.expr
                new_dependencies[resolved_assign.var_name] = resolved_assign.expr
            elif isinstance(stmt, ast.AugAssign):
                resolved_assign = self._parse_function_aug_assign(stmt, dependencies)
                dependencies[resolved_assign.var_name] = resolved_assign.expr
                new_dependencies[resolved_assign.var_name] = resolved_assign.expr
            elif isinstance(stmt, ast.Return):
                return_expr = self._resolve_expr(stmt.value, dependencies)
                if is_function_body and "__RETURN__" in dependencies:
                    # we parsed the level-0 return, so we won't merge it with previous returns as we would have done
                    # when handling a If clause, so we need to update the current else_ field of the return dependency
                    # with this return expression
                    conditional_return = dependencies["__RETURN__"]
                    assert isinstance(conditional_return, Conditional)
                    while conditional_return.else_ is not None:
                        conditional_return = conditional_return.else_
                        assert isinstance(conditional_return, Conditional)
                    conditional_return.else_ = return_expr
                else:
                    new_dependencies["__RETURN__"] = return_expr
            elif isinstance(stmt, ast.If):
                if_else_dependencies = self._parse_if_else(stmt, dependencies)
                dependencies = self._merge_conditional_dependencies(dependencies, if_else_dependencies)  # type: ignore[arg-type]
                new_dependencies = self._merge_conditional_dependencies(
                    new_dependencies, if_else_dependencies  # type: ignore[arg-type]
                )
            else:
                raise NotImplementedError(type(stmt))

        return new_dependencies

    @staticmethod
    def _merge_conditional_dependencies(
        old_dependencies: Dict[str, Expr], new_dependencies: Dict[str, Conditional]
    ) -> Dict[str, Expr]:
        """
        Merge new conditional dependencies with previous dependencies (potentially conditional).
        - If a new dependency does not exist in the old dependencies, simply add it
        - If a new dependency already exist in the old dependencies, two cases:
          -> It is a __RETURN__ dependency that is necessarily conditional,
             thus the else_ field of the old dependency is filled with the new return dependency
          -> Else, if the else_ field of the new dependency is empty, it is filled with the old dependency.
             If the else_ field of the new dependency is already filled, it means the old dependency is obsolete.
        """

        result_dependencies = copy(old_dependencies)
        for dep_name in new_dependencies:
            new_dependency = new_dependencies[dep_name]
            if dep_name in old_dependencies:
                if dep_name == "__RETURN__":
                    # Add the new return Expr in the last nested else_ field of the current return Expr
                    conditional_return = deepcopy(old_dependencies["__RETURN__"])
                    assert isinstance(conditional_return, Conditional)
                    while conditional_return.else_ is not None:
                        conditional_return = conditional_return.else_
                        assert isinstance(conditional_return, Conditional)

                    conditional_return.else_ = new_dependency
                    result_dependencies[dep_name] = conditional_return
                else:
                    if new_dependency.else_ is None:  # Otherwise do nothing, the old dependency is obsolete
                        new_dependency = deepcopy(new_dependency)
                        new_dependency.else_ = old_dependencies[dep_name]
                        result_dependencies[dep_name] = new_dependency
            else:
                # We simply assign to a variable whether it has already been defined before.
                result_dependencies[dep_name] = new_dependency

        return result_dependencies

    def _parse_if_else(self, if_node: ast.If, dependencies: Dict[str, Expr]) -> Dict[str, Conditional]:
        """
        Parse a IF statement, and returns the dependencies created within the if/elif/else blocks.

        Note a ELIF statement is stored as an IF-statement within the `orelse` field of an ast.If object,
        thus this function will be called recursively in case of ELIF statements
        (or obviously in case of nested IF statements).
        """

        condition = self._resolve_expr(if_node.test, dependencies)
        if_dependencies = self._parse_statements(if_node.body, dependencies)
        else_dependencies = self._parse_statements(if_node.orelse, dependencies)

        result_dependencies = {}

        # Add all the dependencies present in the if clause, including common ones with the else clause
        for var_name in if_dependencies:
            # The variable might be defined both in the if and the else
            conditional_or_expr = else_dependencies[var_name] if var_name in else_dependencies else None

            conditional = Conditional(
                condition=condition,
                expr=if_dependencies[var_name],
                else_=conditional_or_expr,
                numpy_alias=self.numpy_alias,
            )
            result_dependencies[var_name] = conditional

        # Add dependencies present in else clause and not in the if clause.
        for var_name in else_dependencies:
            if var_name in result_dependencies:
                continue  # already handled

            # In a if/else case, expr is an Expr, but in a if/elseif case, expr is a Conditional
            conditional = Conditional(
                condition=NegateBooleanExpr(condition),
                expr=else_dependencies[var_name],
                else_=None,
                numpy_alias=self.numpy_alias,
            )
            result_dependencies[var_name] = conditional

        return result_dependencies

    def _parse_function_assign(self, assign_node: ast.Assign, dependencies: Dict[str, Expr]):
        if len(assign_node.targets) > 1:
            raise NotImplementedError("Multiple assignments")

        target = assign_node.targets[0]
        if not isinstance(target, ast.Name):
            raise NotImplementedError("Assignment to a non-name target")

        right = assign_node.value

        expr: Expr = self._resolve_expr(right, dependencies)
        return ResolvedAssign(var_name=target.id, expr=expr)

    def _parse_function_aug_assign(self, aug_assign_node: ast.AugAssign, dependencies: Dict[str, Expr]):
        if not isinstance(aug_assign_node.target, ast.Name):
            raise NotImplementedError("Assignment to a non-name target")

        left = dependencies[aug_assign_node.target.id]
        right = self._resolve_expr(aug_assign_node.value, dependencies)
        expr = BinaryOp(
            left=left,
            right=right,
            pd_symbol=get_op_symbol(aug_assign_node.op),
        )
        return ResolvedAssign(var_name=aug_assign_node.target.id, expr=expr)

    def parse_lambda(self, lambda_node: ast.Lambda) -> Expr:
        parameter_name = lambda_node.args.args[0].arg
        dependencies = {parameter_name: ApplyFuncArg()}
        lambda_expr = self._resolve_expr(lambda_node.body, dependencies)  # type: ignore[arg-type]
        return lambda_expr


def get_package_alias(stmt: ast.stmt, numpy_or_pandas: Literal["numpy", "pandas"]) -> Optional[str]:
    """
    Check whether a statement is a pandas or numpy import and return the corresponding alias.
    Ex:
      - "import pandas" -> pandas
      - "import numpy as np" -> np
      - "import requests as req" -> None
    """

    if not isinstance(stmt, ast.Import):
        return None

    for alias in stmt.names:
        if alias.name == numpy_or_pandas:
            if alias.asname is not None:
                return alias.asname
            return alias.name

    return None


def replace_apply_assignment_inplace(
    lines_of_code: List[str],
    vectorized_code: str,
    apply_start_line: int,
    apply_end_line: int,
) -> None:
    """
    Replace the apply assignment by the new vectorized code. `lines_of_code` is modified inplace.
    """

    lines_of_code[apply_start_line - 1] = vectorized_code
    flag_lines_to_remove_inplace(lines_of_code, apply_start_line, apply_end_line)


def flag_lines_to_remove_inplace(lines_of_code: List[str], start_line: int, end_line: int):
    """
    Set lines between start_line and end_line to None, so they are removed later. We can not remove them directly
    as the previously stored apply_start_line/apply_end_line would not make sense anymore.
    `lines_of_code` is modified inplace.
    """

    for i in range(start_line, end_line):
        lines_of_code[i] = None  # type: ignore


def is_vectorisable_apply_call(call_node: ast.Call) -> bool:
    """
    Check whether the call is a call to `apply`, and tries to determine whether the call
    is performed on DataFrame with axis=0 (implicit or not) in which case we should not vectorize the `apply` call.
    """

    if not (isinstance(call_node.func, ast.Attribute) and call_node.func.attr == "apply"):
        return False

    for keyword in call_node.keywords:
        if keyword.arg == "axis" and isinstance(keyword.value, ast.Constant):  # Dynamic axis parameter unsupported
            if keyword.value.value in [1, "columns"]:
                return True
            if keyword.value.value in [0, "index"]:
                return False

    # Here, we do not know whether `apply` is called on a Series or on DataFrame with implicit axis=0
    # There is no way to be 100% sure of the input caller type, so make the following assumptions:
    #  - Calls like `variable["str"].apply` are most likely Series calls.
    #  - Calling variable names starting with "df" are most likely DataFrame calls.

    if isinstance(call_node.func.value, ast.Subscript):
        return True  # Most likely a Series call

    if isinstance(call_node.func.value, ast.Name) and call_node.func.value.id.startswith("df"):
        return False  # Most likely a DataFrame call

    return True  # Otherwise, assume it is a Series call.


def get_assignment_apply_expr_info(stmt: ast.stmt, input_code: str) -> Optional[ApplyInfo]:
    """
    Check whether a statement is an assignment which right hand side is a call to `apply` method,
    and return related information to the `apply` call.
    """

    if not isinstance(stmt, ast.Assign):
        return None

    call_node = stmt.value
    if not isinstance(call_node, ast.Call):
        return None

    if not is_vectorisable_apply_call(call_node):
        return None

    assert len(stmt.targets) == 1
    assert isinstance(call_node.func, ast.Attribute)

    assigned_var_name = ast.get_source_segment(input_code, stmt.targets[0])
    calling_var_name = ast.get_source_segment(input_code, call_node.func.value)

    # 'func' parameter is either passed as kwarg or arg.
    for keyword in call_node.keywords:
        if keyword.arg == "func":
            func_expr = keyword.value
            break
    else:
        func_expr = call_node.args[0]

    assert assigned_var_name is not None and calling_var_name is not None
    assert stmt.end_lineno is not None

    return ApplyInfo(
        assigned_var_name=assigned_var_name,
        calling_var_name=calling_var_name,
        start_lineno=stmt.lineno,
        end_lineno=stmt.end_lineno,
        ast_func_expr=func_expr,
    )


def try_replace_apply_lambda_inplace(
    lines_of_code: List[str], apply_info: ApplyInfo, parser: FunctionBodyParser
) -> bool:
    """
    Resolve a lambda expression within an `apply` call, and replace the corresponding assignment
    with vectorized code if possible. Return a boolean indicating whether some code has been replaced or not.
    """

    assert isinstance(apply_info.ast_func_expr, ast.Lambda)

    try:
        expr = parser.parse_lambda(apply_info.ast_func_expr)
    except NotImplementedError:
        return False

    vectorized_code = convert_expr_to_vectorized_code(expr, apply_info.assigned_var_name, apply_info.calling_var_name)
    replace_apply_assignment_inplace(lines_of_code, vectorized_code, apply_info.start_lineno, apply_info.end_lineno)

    return True


def try_replace_apply_func_inplace(
    lines_of_code: List[str], apply_info: ApplyInfo, func_name: str, parser: FunctionBodyParser
):
    """
    Build a lambda expression for a func name (eg: apply(pd.isna) -> apply(lambda x: pd.isna(x)), resolve this
    lambda expression and replace the apply assignment by vectorized operations if possible.
    Return a boolean indicating whether some code has been replaced or not.
    """

    lambda_str = f"lambda x: {func_name}(x)"
    expr_node = ast.parse(lambda_str).body[0]
    assert isinstance(expr_node, ast.Expr)
    apply_info.ast_func_expr = expr_node.value

    return try_replace_apply_lambda_inplace(lines_of_code, apply_info, parser)


def replace_apply(input_code: str):

    statements = ast.parse(input_code).body
    parser = FunctionBodyParser(input_code)

    # Look either for pandas/numpy imports, or for assignment containing `apply` calls.
    lines = input_code.split("\n")
    for stmt in statements:
        if pandas_alias := get_package_alias(stmt, "pandas"):  # eg: import pandas as pd
            parser.pandas_alias = pandas_alias
        elif numpy_alias := get_package_alias(stmt, "numpy"):  # eg: import numpy as np
            parser.numpy_alias = numpy_alias
        elif isinstance(stmt, ast.FunctionDef):
            parser.udf_name_to_func_def_node[stmt.name] = stmt
        elif apply_info := get_assignment_apply_expr_info(stmt, input_code):  # eg: s = s.apply(...)
            if isinstance(apply_info.ast_func_expr, ast.Lambda):
                _ = try_replace_apply_lambda_inplace(lines, apply_info, parser)
            elif isinstance(apply_info.ast_func_expr, ast.Attribute):
                func_name = ast.get_source_segment(input_code, apply_info.ast_func_expr)
                _ = try_replace_apply_func_inplace(lines, apply_info, func_name, parser)  # type: ignore[arg-type]
            elif isinstance(apply_info.ast_func_expr, ast.Name):
                func_name = apply_info.ast_func_expr.id
                has_replaced = try_replace_apply_func_inplace(lines, apply_info, func_name, parser)
                # Remove the UDF definition
                if has_replaced and func_name in parser.udf_name_to_func_def_node:
                    udf_def_node = parser.udf_name_to_func_def_node[func_name]
                    assert udf_def_node.end_lineno is not None
                    flag_lines_to_remove_inplace(lines, udf_def_node.lineno - 1, udf_def_node.end_lineno)

    # Remove lines corresponding to functions definitions. We remove them at the end
    # otherwise apply locations would not be relevant anymore.
    lines = [line for line in lines if line is not None]
    output_code = "\n".join(lines)

    if should_import_numpy(output_code, parser.numpy_alias):
        numpy_import = "import numpy" if parser.numpy_alias == "numpy" else f"import numpy as {parser.numpy_alias}"
        output_code = numpy_import + "\n" + output_code

    return output_code


def should_import_numpy(output_code: str, numpy_alias: str) -> bool:
    """
    We might introduce a numpy dependency (`np.select`) when we meet conditional statements,
    thus we should import numpy in this case if it is not already imported.
    """

    if f"{numpy_alias}.select" not in output_code:
        return False
    if numpy_alias == "numpy" and "import numpy" not in output_code:
        return True
    if numpy_alias != "numpy" and f"import numpy as {numpy_alias}" not in output_code:
        return True
    return False

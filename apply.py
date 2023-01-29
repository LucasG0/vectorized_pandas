from __future__ import annotations

import ast
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from expression import (
    ApplyFuncArg,
    BinaryBooleanExpr,
    BinaryOp,
    Conditional,
    Constant,
    Expr,
    NegateBooleanExpr,
    TrivialFuncCall,
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
    if isinstance(operator, ast.Gt):
        return ">"
    if isinstance(operator, ast.GtE):
        return ">="
    if isinstance(operator, ast.Lt):
        return "<"
    if isinstance(operator, ast.LtE):
        return "<="
    raise NotImplementedError(type(operator))


# Default packages aliases if the input code omits the import line.
DEFAULT_PANDAS_ALIAS = ["pandas", "pd"]
DEFAULT_NUMPY_ALIAS = ["numpy", "np"]

REPLACEABLE_PANDAS_FUNCS = {"isna", "isnull"}
REPLACEABLE_NUMPY_FUNCS = {"isnan"}


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

    pandas_alias: List[str]
    numpy_alias: List[str]

    def __init__(self):
        self.pandas_alias = DEFAULT_PANDAS_ALIAS
        self.numpy_alias = DEFAULT_NUMPY_ALIAS

    def is_replaceable_func(self, alias, func_name) -> bool:
        return (alias in self.pandas_alias and func_name in REPLACEABLE_PANDAS_FUNCS) or (
            alias in self.numpy_alias and func_name in REPLACEABLE_NUMPY_FUNCS
        )

    def _resolve_expr(self, expr_node: Optional[ast.expr], dependencies: Dict[str, Expr]) -> Expr:
        """
        Build an Expr object from a ast.expr node, using the current dependencies.
        """

        assert expr_node is not None, "Trying to resolve an empty expression node"

        # TODO Handle the fact that if left and/or right are Conditional, we should build a Conditional binary op.
        if isinstance(expr_node, ast.BinOp):
            left = self._resolve_expr(expr_node.left, dependencies)
            right = self._resolve_expr(expr_node.right, dependencies)
            return BinaryOp(left=left, right=right, pd_symbol=get_op_symbol(expr_node.op))

        if isinstance(expr_node, ast.Compare):
            assert len(expr_node.comparators) == 1, "multiple comparators not supported"
            assert len(expr_node.ops) == 1, "multiple comparison ops not supported"
            left = self._resolve_expr(expr_node.left, dependencies)
            right = self._resolve_expr(expr_node.comparators[0], dependencies)
            return BinaryBooleanExpr(
                left=left,
                right=right,
                pd_symbol=get_comp_symbol(expr_node.ops[0]),
            )

        if isinstance(expr_node, ast.Constant):
            return Constant(expr_node.value)

        if isinstance(expr_node, ast.Name):
            return dependencies[expr_node.id]  # The expression is supposed to be already resolved ?

        if isinstance(expr_node, ast.Call):
            if not isinstance(expr_node.func, ast.Attribute) or not isinstance(expr_node.func.value, ast.Name):
                raise NotImplementedError(f"Function call node {expr_node}")

            alias = expr_node.func.value.id
            func_name = expr_node.func.attr
            if self.is_replaceable_func(expr_node.func.value.id, expr_node.func.attr):
                return TrivialFuncCall(
                    alias=alias,
                    func_name=func_name,
                    params=[self._resolve_expr(arg, dependencies) for arg in expr_node.args],
                )

        raise NotImplementedError(type(expr_node))

    def parse_function_def(self, func_def_node: ast.FunctionDef) -> Expr:
        """
        Entry point of this class to parse a function passed to `apply`. Initiate the parsing of the function body,
        and return the Expr object associated to the __RETURN__ dependency. See details in `parse_function_body` docstring.
        """

        dependencies: Dict[str, Expr] = {}
        parameter_name = func_def_node.args.args[0].arg
        dependencies[parameter_name] = ApplyFuncArg()

        dependencies = self.parse_function_body(func_def_node.body, dependencies)

        return dependencies["__RETURN__"]

    def parse_function_body(self, body: List[ast.stmt], dependencies: Dict[str, Expr]) -> Dict[str, Expr]:
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
                resolved_assign = self.parse_function_assign(stmt, dependencies)
                dependencies[resolved_assign.var_name] = resolved_assign.expr
                new_dependencies[resolved_assign.var_name] = resolved_assign.expr
            elif isinstance(stmt, ast.AugAssign):
                resolved_assign = self.parse_function_aug_assign(stmt, dependencies)
                dependencies[resolved_assign.var_name] = resolved_assign.expr
                new_dependencies[resolved_assign.var_name] = resolved_assign.expr
            elif isinstance(stmt, ast.Return):
                # If a return dependency already exists, it must be a Conditional with an empty `else_` field
                # (otherwise the code would terminate). So we fill the `else` field of the existing return dependency
                # with the current return expr.
                return_expr = self._resolve_expr(stmt.value, dependencies)
                if "__RETURN__" not in dependencies:
                    dependencies["__RETURN__"] = return_expr
                    new_dependencies["__RETURN__"] = return_expr
                else:
                    conditional_return = dependencies["__RETURN__"]
                    assert isinstance(conditional_return, Conditional)
                    while conditional_return.else_ is not None:
                        conditional_return = conditional_return.else_
                        assert isinstance(conditional_return, Conditional)

                    conditional_return.else_ = return_expr
                    # TODO probably safer to also retrieve the conditional_return from "new_dependencies"
                    #  and to set it's else_ field as the one from "dependencies" is not necessarily
                    #  the same than in "new_dependencies" (who knows what happens in "merge_dependencies" ?)

            elif isinstance(stmt, ast.If):
                if_else_dependencies = self.parse_if_else(stmt, dependencies)
                dependencies = self.merge_dependencies(dependencies, if_else_dependencies)  # type: ignore[arg-type]
                new_dependencies = self.merge_dependencies(new_dependencies, if_else_dependencies)  # type: ignore[arg-type]
            else:
                raise NotImplementedError(type(stmt))

        return new_dependencies

    @staticmethod
    def merge_dependencies(old_dependencies: Dict[str, Expr], new_dependencies: Dict[str, Expr]) -> Dict[str, Expr]:
        """
        For a common dependency between the old ones and the new ones:
        - If the new dependency is a conditional and the else is empty, the else becomes the old dependency
        - Otherwise, simply override the old dependency by the new one
        """

        result_dependencies = copy(old_dependencies)
        for dep_name in new_dependencies:
            new_dependency = new_dependencies[dep_name]
            if dep_name in old_dependencies and isinstance(new_dependency, Conditional):
                if new_dependency.else_ is None:  # Otherwise do nothing, the old dependency is obsolete
                    new_dependency = deepcopy(new_dependency)
                    new_dependency.else_ = old_dependencies[dep_name]
                    result_dependencies[dep_name] = new_dependency
            else:
                # We simply assign to a variable whether it has already been defined before.
                result_dependencies[dep_name] = new_dependency

        return result_dependencies

    def parse_if_else(self, if_node: ast.If, dependencies: Dict[str, Expr]) -> Dict[str, Conditional]:
        """
        Parse a IF statement, and returns the dependencies created within the if/elif/else blocks.

        Note a ELIF statement is stored as an IF-statement within the `orelse` field of an ast.If object,
        thus this function will be called recursively in case of ELIF statements
        (or obviously in case of nested IF statements).
        """

        condition = self._resolve_expr(if_node.test, dependencies)
        if_dependencies = self.parse_function_body(if_node.body, dependencies)
        else_dependencies = self.parse_function_body(if_node.orelse, dependencies)

        result_dependencies = {}

        # Add all the dependencies present in the if clause, including common ones with the else clause
        for var_name in if_dependencies:
            # The variable might be defined both in the if and the else
            conditional_or_expr = else_dependencies[var_name] if var_name in else_dependencies else None

            conditional = Conditional(
                condition=condition,
                expr=if_dependencies[var_name],
                else_=conditional_or_expr,
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
            )
            result_dependencies[var_name] = conditional

        return result_dependencies

    def parse_function_assign(self, assign_node: ast.Assign, dependencies: Dict[str, Expr]):
        if len(assign_node.targets) > 1:
            raise NotImplementedError("Multiple assignments")

        target = assign_node.targets[0]
        if not isinstance(target, ast.Name):
            raise NotImplementedError("Assignment to a non-name target")

        right = assign_node.value

        expr: Expr = self._resolve_expr(right, dependencies)
        return ResolvedAssign(var_name=target.id, expr=expr)

    def parse_function_aug_assign(self, aug_assign_node: ast.AugAssign, dependencies: Dict[str, Expr]):
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


def get_assignment_apply_expr_info(stmt: ast.stmt, input_code: str) -> Optional[ApplyInfo]:
    """
    Check whether a statement is an assignment which right hand side is a call to `apply` method,
    and return related information to the `apply` call.
    """

    if not isinstance(stmt, ast.Assign):
        return None

    right = stmt.value
    if not isinstance(right, ast.Call):
        return None

    if not (isinstance(right.func, ast.Attribute) and right.func.attr == "apply"):
        return None

    assert len(stmt.targets) == 1

    assigned_var_name = ast.get_source_segment(input_code, stmt.targets[0])
    calling_var_name = ast.get_source_segment(input_code, right.func.value)

    for keyword in right.keywords:
        if keyword.arg == "func":
            func_expr = keyword.value
            break
    else:
        func_expr = right.args[0]

    assert assigned_var_name is not None and calling_var_name is not None
    assert stmt.end_lineno is not None

    return ApplyInfo(
        assigned_var_name=assigned_var_name,
        calling_var_name=calling_var_name,
        start_lineno=stmt.lineno,
        end_lineno=stmt.end_lineno,
        ast_func_expr=func_expr,
    )


def maybe_replace_apply_lambda_inplace(lines_of_code: List[str], apply_info: ApplyInfo, parser: FunctionBodyParser):
    """
    Resolve a lambda expression within an `apply` call, and replace the corresponding assignment
    with vectorized code if possible.
    """

    assert isinstance(apply_info.ast_func_expr, ast.Lambda)
    try:
        expr = parser.parse_lambda(apply_info.ast_func_expr)
    except NotImplementedError:
        return

    vectorized_code = convert_expr_to_vectorized_code(expr, apply_info.assigned_var_name, apply_info.calling_var_name)
    replace_apply_assignment_inplace(lines_of_code, vectorized_code, apply_info.start_lineno, apply_info.end_lineno)


def maybe_replace_apply_package_func_inplace(
    lines_of_code: List[str],
    apply_info: ApplyInfo,
    parser: FunctionBodyParser,
):
    """
    Check whether the external package function called within `apply` is vectorizable,
    and replace the corresponding assignment with vectorized code if possible.
    """

    assert isinstance(apply_info.ast_func_expr, ast.Attribute)

    func_node = apply_info.ast_func_expr
    if not isinstance(func_node.value, ast.Name):
        return

    alias = func_node.value.id
    func_name = func_node.attr
    if not parser.is_replaceable_func(alias, func_name):
        return

    vectorized_code = f"{apply_info.assigned_var_name} = {alias}.{func_name}({apply_info.calling_var_name})"
    replace_apply_assignment_inplace(lines_of_code, vectorized_code, apply_info.start_lineno, apply_info.end_lineno)


def maybe_replace_apply_udf_inplace(
    lines_of_code: List[str],
    func_def_node: ast.FunctionDef,
    apply_info: ApplyInfo,
    parser: FunctionBodyParser,
):
    """
    Check whether the user defined function called within `apply` is vectorizable,
    and replace the corresponding assignment with vectorized code if possible.
    """

    try:
        expr = parser.parse_function_def(func_def_node)
    except NotImplementedError:
        return

    vectorized_code = convert_expr_to_vectorized_code(
        expr=expr,
        calling_var_name=apply_info.calling_var_name,
        assigned_var_name=apply_info.assigned_var_name,
    )

    replace_apply_assignment_inplace(lines_of_code, vectorized_code, apply_info.start_lineno, apply_info.end_lineno)

    assert func_def_node.end_lineno is not None
    flag_lines_to_remove_inplace(lines_of_code, func_def_node.lineno - 1, func_def_node.end_lineno)


def replace_apply(input_code: str):
    """
    - Build the ast of input_code.
    - Look for `apply` calls.
      -> If the function applied is a lambda or a pandas/numpy function, it is directly replaced.
      -> Otherwise the function called is stored, and its body is parsed afterwards.
    - Parse the applied functions bodies, and generate the new code.
    """

    statements = ast.parse(input_code).body

    # Store funcs info used in apply: func_name -> (assigned_var_name, calling_var_name, apply_lineno, apply_end_lineno)
    func_name_to_apply_info: Dict[str, ApplyInfo] = {}

    parser = FunctionBodyParser()

    # Look either for pandas/numpy imports, or for assignment containing `apply` calls.
    lines = input_code.split("\n")
    for stmt in statements:
        if pandas_alias := get_package_alias(stmt, "pandas"):  # eg: import pandas as pd
            parser.pandas_alias = [pandas_alias]
        elif numpy_alias := get_package_alias(stmt, "numpy"):  # eg: import numpy as np
            parser.numpy_alias = [numpy_alias]
        elif apply_info := get_assignment_apply_expr_info(stmt, input_code):  # eg: s = s.apply(...)
            if isinstance(apply_info.ast_func_expr, ast.Lambda):  # eg: apply(lambda x: x+1)
                maybe_replace_apply_lambda_inplace(lines, apply_info, parser)
            elif isinstance(apply_info.ast_func_expr, ast.Attribute):  # eg: apply(pd.isna)
                maybe_replace_apply_package_func_inplace(lines, apply_info, parser)
            elif isinstance(apply_info.ast_func_expr, ast.Name):  # eg: apply(udf)
                # Keep the name of the udf applied, and we will resolve the function return expression afterwards.
                func_name_to_apply_info[apply_info.ast_func_expr.id] = apply_info

    # Resolve the statements of functions applied, and replace the `apply` accordingly if possible.
    if len(func_name_to_apply_info) > 0:
        for stmt in statements:
            if isinstance(stmt, ast.FunctionDef) and stmt.name in func_name_to_apply_info:
                maybe_replace_apply_udf_inplace(lines, stmt, func_name_to_apply_info[stmt.name], parser)

    # Remove lines corresponding to functions definitions. We remove them at the end
    # otherwise apply locations would not be relevant anymore.
    lines = [line for line in lines if line is not None]

    result = "\n".join(lines)
    return result

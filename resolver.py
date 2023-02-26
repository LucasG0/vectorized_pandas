from __future__ import annotations

import ast
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

from constants import (
    DEFAULT_NUMPY_ALIAS,
    DEFAULT_PANDAS_ALIAS,
    NATIVE_TYPE_FUNCS,
    NUMPY_TYPE_FUNCS,
    REPLACEABLE_NUMPY_FUNCS,
    REPLACEABLE_PANDAS_FUNCS,
    STR_METHODS,
)
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
    Global,
    NegateBooleanExpr,
    StrMethodCall,
    TrivialFuncCall,
    TypeCastFuncCall,
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


@dataclass
class ResolvedAssign:
    var_name: str
    expr: Expr


class ExpressionResolver:
    """
    Purpose is to build the corresponding vectorized code at the Series/DataFrame level from a function
    being applied.
    """

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
        if isinstance(attr, ast.Name) and attr.id in dependencies and isinstance(dependencies[attr.id], ApplyFuncArg):
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

    def _resolve_if_expression(self, if_exp_node: ast.IfExp, dependencies: Dict[str, Expr]) -> Conditional:
        return Conditional(
            expr=self._resolve_expr(if_exp_node.body, dependencies),
            condition=self._resolve_expr(if_exp_node.test, dependencies),
            else_=self._resolve_expr(if_exp_node.orelse, dependencies),
            numpy_alias=self.numpy_alias,
        )

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
                global_dependencies = {
                    var_name: dependencies[var_name]
                    for var_name in dependencies
                    if isinstance(dependencies[var_name], Global)
                }
                return self._parse_function_def(
                    self.udf_name_to_func_def_node[func.id], args_expr, global_dependencies  # type:ignore[arg-type]
                )

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

        if isinstance(expr_node, ast.IfExp):
            return self._resolve_if_expression(expr_node, dependencies)

        raise NotImplementedError(f"{expr_node}=")

    def _parse_function_def(
        self, func_def_node: ast.FunctionDef, args_expr: List[Expr], global_dependencies: Dict[str, Global]
    ) -> Expr:
        """
        Initiate the parsing of the function body, and return the Expr object associated to the __RETURN__ dependency.
        See details in `parse_function_body` docstring.
        """

        args = func_def_node.args.args
        assert len(args) == len(args_expr)
        args_dependencies = dict(zip([arg.arg for arg in args], args_expr))
        dependencies: Dict[str, Expr] = global_dependencies | args_dependencies  # type: ignore
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

    def parse_lambda_start(self, lambda_node: ast.Lambda, global_dependencies: Dict[str, Global]) -> Expr:
        parameter_name = lambda_node.args.args[0].arg
        dependencies: Dict[str, Expr] = copy(global_dependencies)  # type:ignore[arg-type]
        dependencies[parameter_name] = ApplyFuncArg()
        lambda_expr = self._resolve_expr(lambda_node.body, dependencies)
        return lambda_expr

    def parse_global_variable(
        self, assign_node: ast.Assign, global_dependencies: Dict[str, Global]
    ) -> Optional[Global]:
        """
        Try to resolve a global variable and if it success, store the variable name into a Global expr.
        Global behavior is different because we want to refer to the variable name instead of the
        resolved expr.
        """

        try:
            self._resolve_expr(assign_node.value, dependencies=global_dependencies)  # type: ignore[arg-type]
        except NotImplementedError:
            return None

        if len(assign_node.targets) > 1:
            raise NotImplementedError("Multiple assignments")

        target = assign_node.targets[0]
        if not isinstance(target, ast.Name):
            raise NotImplementedError("Assignment to a non-name target")

        return Global(target.id)

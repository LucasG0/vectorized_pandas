from __future__ import annotations

import ast
from typing import Dict

from black import FileMode, format_str

from expression import Global, convert_expr_to_expr_node
from resolver import ExpressionResolver


class ApplyReplacer(ast.NodeVisitor):
    """
    Main purpose is to find apply calls and to replace them by vectorized expressions.
    Also detect numpy/pandas alias used.
    """

    parser: ExpressionResolver
    input_code: str
    global_dependencies: Dict[str, Global]

    def __init__(self, parser: ExpressionResolver, input_code: str):
        self.parser = parser
        self.input_code = input_code
        self.global_dependencies = {}

    def visit_Import(self, import_node: ast.Import):
        """
        Look for pandas/numpy imports and set the corresponding aliases.
        Ex:
          - "import pandas" -> pandas
          - "import numpy as np" -> np
          - "import requests as req" -> None
        """

        for alias in import_node.names:
            if alias.name == "pandas":
                self.parser.pandas_alias = alias.asname if alias.asname is not None else alias.name

            if alias.name == "numpy":
                self.parser.numpy_alias = alias.asname if alias.asname is not None else alias.name

        self.generic_visit(import_node)

    def visit_Assign(self, assign_node: ast.Assign):
        """
        Register global variables.
        """

        if isinstance(assign_node.parent, ast.Module):  # type:ignore[attr-defined]
            global_ = self.parser.parse_global_variable(assign_node, global_dependencies=self.global_dependencies)
            if global_ is not None:
                self.global_dependencies[global_.name] = global_

        self.generic_visit(assign_node)

    def visit_FunctionDef(self, func_def_node: ast.FunctionDef):
        self.parser.udf_name_to_func_def_node[func_def_node.name] = func_def_node
        self.generic_visit(func_def_node)

    def _build_resolvable_expr_from_func_arg(self, func_arg_node: ast.expr) -> ast.Lambda:
        if isinstance(func_arg_node, ast.Lambda):
            return func_arg_node

        if isinstance(func_arg_node, (ast.Name, ast.Attribute)):
            func_name = ast.get_source_segment(self.input_code, func_arg_node)
            lambda_str = f"lambda x: {func_name}(x)"
            expr_node = ast.parse(lambda_str).body[0]
            assert isinstance(expr_node, ast.Expr)

            lambda_node = expr_node.value
            assert isinstance(lambda_node, ast.Lambda)

            return lambda_node

        raise NotImplementedError(f"{type(func_arg_node)=}")

    @staticmethod
    def _replace_ast_node_inplace(old_node: ast.AST, new_node: ast.AST):

        parent = old_node.parent  # type: ignore[attr-defined]
        for field, value in ast.iter_fields(parent):
            if value is old_node:
                parent.__setattr__(field, new_node)
                return

            if isinstance(value, list):
                value[:] = [new_node if child is old_node else child for child in value]

    def _insert_temp_assignment_inplace(self, call_node: ast.Call, call_str: str) -> str:
        """
        If the `apply` call is located within a chained assignment, we don't want to produce vectorized code
        that performs the pre-apply operations multiple times. Thus, we introduce a assignment to a temporary
        variable that will be reused within the vectorized expression.
        """

        # Build the assignment node
        temp_var = "s_temp"
        assign_stmt = ast.parse(f"{temp_var} = {call_str}").body[0]

        # Find the parent statement that contains the call
        curr_node = call_node
        while not isinstance(curr_node.parent, ast.stmt):  # type:ignore[attr-defined]
            curr_node = curr_node.parent  # type:ignore[attr-defined]
        stmt_node = curr_node.parent  # type:ignore[attr-defined]

        # Insert the new assignment just above the statement containing the apply
        # Usually the parent node has an attribute "body" containing the list of statements,
        # but for ast.If the statement might also be in the orelse field.
        try:
            assert hasattr(stmt_node.parent, "body")  # type:ignore[attr-defined]
            stmt_list = stmt_node.parent.body  # type:ignore[attr-defined]
            index_stmt = stmt_list.index(stmt_node)
        except ValueError:
            assert hasattr(stmt_node.parent, "orelse")  # type:ignore[attr-defined]
            stmt_list = stmt_node.parent.orelse  # type:ignore[attr-defined]
            index_stmt = stmt_list.index(stmt_node)

        stmt_list.insert(index_stmt, assign_stmt)
        return temp_var

    def visit_Call(self, call_node: ast.Call):
        """
        Identify whether a call is an `apply` call, and replace it by a vectorized expression if possible.
        This method might modify the ast inplace in the 3 following ways:
        - Replace the `apply` call node
        - Drop the UDF definition
        - Introduce temporary variable assignment.
        """

        if not self._is_vectorisable_apply_call(call_node):
            self.generic_visit(call_node)
            return

        assert isinstance(call_node.func, ast.Attribute)

        func_arg_node = self._get_func_arg_node(call_node)
        lambda_expr_to_resolve = self._build_resolvable_expr_from_func_arg(func_arg_node)

        calling_code = ast.get_source_segment(self.input_code, call_node.func.value)
        assert calling_code is not None
        if isinstance(call_node.func.value, ast.Call):
            calling_code = self._insert_temp_assignment_inplace(call_node, calling_code)

        try:
            resolved_expr = self.parser.parse_lambda_start(lambda_expr_to_resolve, self.global_dependencies)
        except TypeError:
            self.generic_visit(call_node)
            return

        new_node = convert_expr_to_expr_node(resolved_expr, calling_code)
        self._replace_ast_node_inplace(call_node, new_node)

        # Drop the udf function definition
        if isinstance(func_arg_node, ast.Name) and func_arg_node.id in self.parser.udf_name_to_func_def_node:
            func_def_node = self.parser.udf_name_to_func_def_node[func_arg_node.id]
            func_def_node.parent.body.remove(func_def_node)  # type:ignore[attr-defined]

        self.generic_visit(new_node)

    @staticmethod
    def _is_vectorisable_apply_call(call_node: ast.Call) -> bool:
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

    def _get_func_arg_node(self, call_node: ast.Call) -> ast.expr:
        assert isinstance(call_node.func, ast.Attribute)

        # 'func' parameter is either passed as kwarg or arg.
        for keyword in call_node.keywords:
            if keyword.arg == "func":
                func_arg_node = keyword.value
                break
        else:
            func_arg_node = call_node.args[0]

        return func_arg_node


def replace_apply(input_code: str):
    """
    # TODO doc
    """

    parser = ExpressionResolver(input_code)
    ast_root = ast.parse(input_code)

    for node in ast.walk(ast_root):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]

    visitor = ApplyReplacer(parser=parser, input_code=input_code)

    # This call modifies ast_root inplace if some apply calls can be vectorized
    visitor.visit(ast_root)
    output_code = ast.unparse(ast_root)

    if should_import_numpy(output_code, parser.numpy_alias):
        numpy_import = "import numpy" if parser.numpy_alias == "numpy" else f"import numpy as {parser.numpy_alias}"
        output_code = numpy_import + "\n" + output_code

    # black formatting
    output_code = format_str(output_code, mode=FileMode(line_length=120))
    assert output_code[-1] == "\n"
    output_code = output_code[:-1]

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

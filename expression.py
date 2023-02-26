from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

TO_REPLACE_BY_VAR_NAME = "__TO_REPLACE_BY_VAR_NAME__"


@dataclass
class Expr:
    """
    Abstract class representing a python expression.
    """

    def to_vectorized_code(self) -> str:
        """
        Convert the python expression into pandas vectorized code. This method is meant to be called
        on the return expression of a function passed to `apply` or on a lambda expression.
        """

        raise NotImplementedError("Abstract method")


@dataclass
class Global(Expr):
    """
    Only need variable name as we want to refer to the variable name instead of the
    resolved expr.
    """

    name: str
    # resolved_expr: Expr  #  Not needed atm but might be useful later ?

    def to_vectorized_code(self) -> str:
        return self.name


@dataclass
class BinaryOp(Expr):
    left: Expr
    right: Expr
    pd_symbol: str

    def to_vectorized_code(self) -> str:
        return "(" + self.left.to_vectorized_code() + " " + self.pd_symbol + " " + self.right.to_vectorized_code() + ")"


@dataclass
class Constant(Expr):
    value: Any

    def to_vectorized_code(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)


@dataclass
class Alias(Expr):
    name: str

    def to_vectorized_code(self) -> str:
        return self.name


@dataclass
class FString(Expr):
    items: List[Expr]

    def _format_item_code(self, item_code: str) -> str:
        if TO_REPLACE_BY_VAR_NAME in item_code:
            return f"{item_code}.astype(str)"
        if not item_code.startswith('"') and not item_code.startswith("'"):
            return f"str({item_code})"
        return item_code

    def to_vectorized_code(self) -> str:
        """
        Concatenate all expressions within the fstring, and perform an explicit cast to string dtype,
        either by using `.astype(str)` for series expressions, or using `str(...)` for non-string scalars.
        We assume as of writing that we can identify a series expression by checking
        whether it contains a reference to the function parameter, ie TO_REPLACE_BY_VAR_NAME.
        """

        item_codes = [item.to_vectorized_code() for item in self.items]
        item_codes = [self._format_item_code(item_code) for item_code in item_codes]

        return " + ".join(item_codes)


@dataclass
class ApplyFuncArg(Expr):
    """
    Identifies the input parameter of the function applied. We need to differentiate it from regular variables
    as it will be converted to the variable name calling `apply` when vectorizing.

    eg:
    def func(row_scalar):  # Identifies 'row_scalar'
      return row_scalar + 1
    """

    def to_vectorized_code(self) -> str:
        """
        A function return expression should be resolvable without knowing the context of where it is called, thus
        here we ignore the variable name calling `apply`, so we put a sentinel value that will be replace afterwards.
        """

        return TO_REPLACE_BY_VAR_NAME


@dataclass
class ApplyFuncArgColumn(Expr):
    """
    Identifies a column access from the input row parameter of the function applied. We need to differentiate it from regular variables
    as it will be converted to the variable name with corresponding column access calling `apply` when vectorizing.

    eg:
    def func(row):
      return row["a"] + 1  # identifies row["a"]
    """

    column_name: str
    dot_notation: bool

    def to_vectorized_code(self) -> str:
        """
        A function return expression should be resolvable without knowing the context of where it is called, thus
        here we ignore the variable name calling `apply`, so we put a sentinel value that will be replace afterwards.
        """

        if self.dot_notation:
            return TO_REPLACE_BY_VAR_NAME + "." + self.column_name
        return TO_REPLACE_BY_VAR_NAME + "[" + self.column_name + "]"


@dataclass
class Conditional(Expr):
    """
    Represent an expression which value depends on at least one condition. If the expression depends on
    multiple conditions, typically due to a if/elseif/else clauses, this is represented by a first Conditional object
    holding the if condition as `condition` field, the corresponding expressions `expr` field, and with a nested
    Conditional object as the else_ clause. In this way, multiple elseif/else clauses are represented by multiple
    nested Conditional objects.

    A Conditional expression is meant to be vectorized using a `np.select(conditions=..., choices=..., default=...)`.
    """

    expr: Expr
    condition: Expr
    else_: Optional[Union[Conditional, Expr]]  # ie elsif/else/nothing
    numpy_alias: str

    def get_conditions_and_choices_and_default(self) -> Tuple[List[str], List[str], str]:
        """
        Build formatted conditions, choices and default in order to build the vectorized code using `np.select`.
        This method is called recursively on the nested Conditional objects on the `else_` field in order to merge
        the nested formatted conditions with the condition of the current Conditional object.

        eg:
            def func(val):
                if val == 1:
                    return "B"
                elif val == 2:
                    return "C"
                return val
            s = s.apply(func)

            would be converted to

            s = np.select(conditions=[(s == 1), (s == 2)], choices=["B", "C"], default=s)

            Note the second condition is NOT (s != 1) & (s == 2) as np.select matches the first
            condition verified within the list of conditions.
        """

        # As of writing only a non-Conditional `condition` field is supported,
        # but the expression and else_ fields can be Conditional.
        assert not isinstance(self.condition, Conditional)

        condition = self.condition.to_vectorized_code()

        if isinstance(self.expr, Conditional):  # "terminal" expr
            (
                expr_conditions,
                expr_choices,
                expr_default,
            ) = self.expr.get_conditions_and_choices_and_default()
            conditions = [condition + " & " + cond for cond in expr_conditions] + [condition]
            choices = expr_choices + [expr_default]
        else:
            conditions = [condition]
            choices = [self.expr.to_vectorized_code()]

        if self.else_ is None:
            default = "None"  # rare case corresponding to the implicit return None of a function. TODO test it
        elif isinstance(self.else_, Conditional):
            (
                else_conditions,
                else_choices,
                default,
            ) = self.else_.get_conditions_and_choices_and_default()
            conditions += [cond for cond in else_conditions]
            choices += else_choices
        else:  # "terminal" expression
            default = self.else_.to_vectorized_code()

        return conditions, choices, default

    def to_vectorized_code(self) -> str:
        """
        Build a np.select expression from the current Conditional object.
        """

        conditions, choices, default = self.get_conditions_and_choices_and_default()
        result = f"{self.numpy_alias}.select(conditions=[{', '.join(conditions)}], choices=[{', '.join(choices)}], default={default})"
        return result


@dataclass
class BinaryBooleanExpr(BinaryOp):
    pass


@dataclass
class NegateBooleanExpr(Expr):
    expr: Expr

    def to_vectorized_code(self) -> str:
        return "~" + self.expr.to_vectorized_code()


@dataclass
class TrivialFuncCall(Expr):
    """
    Represent a function call of a function that can act on both scalar and series. eg: pd.isna.
    """

    alias: str
    func_name: str
    params: List[Expr]

    def to_vectorized_code(self) -> str:
        formatted_params = [remove_extra_parenthesis(param.to_vectorized_code()) for param in self.params]
        return self.alias + "." + self.func_name + "(" + ",".join(formatted_params) + ")"


@dataclass
class StrMethodCall(Expr):
    """
    Represent a call to a str method on a scalar to convert to the corresponding pandas string accessor method.
    eg: val.upper() -> s.str.upper()
    """

    caller: Expr
    method_name: str
    params: List[Expr]  # assumed constant

    def to_vectorized_code(self) -> str:
        params_code = ",".join([param.to_vectorized_code() for param in self.params])
        return f"{self.caller.to_vectorized_code()}.str.{self.method_name}({params_code})"


@dataclass
class TypeCastFuncCall(Expr):
    """
    Represent a native or a numpy type cast. eg: int(x), np.float64(x).
    """

    alias: Optional[str]
    func_name: str
    param: Expr  # do not support extra params that could be passed to np.int32 for instance

    def to_vectorized_code(self) -> str:
        formatted_param = remove_extra_parenthesis(self.param.to_vectorized_code())
        full_func_name = self.alias + "." + self.func_name if self.alias is not None else self.func_name
        return formatted_param + ".astype(" + full_func_name + ")"


def remove_extra_parenthesis(expr_code: str) -> str:
    """
    Resolving binary/conditional expressions might have added extra paranthesis around expressions
    that do not need them. This function removes them.

    eg: (s + 1) -> s + 1
    """

    if expr_code.startswith("(") and expr_code.endswith(")"):
        return expr_code[1:-1]
    return expr_code


def convert_expr_to_expr_node(expr: Expr, calling_var_name: str) -> ast.expr:

    expr_code = expr.to_vectorized_code()
    expr_code = expr_code.replace(TO_REPLACE_BY_VAR_NAME, calling_var_name)
    if isinstance(expr, (BinaryOp, BinaryBooleanExpr)):
        expr_code = remove_extra_parenthesis(expr_code)

    expr_node = ast.parse(expr_code).body[0]
    assert isinstance(expr_node, ast.Expr)
    return expr_node.value

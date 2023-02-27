import ast


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
    "nan_to_num",
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

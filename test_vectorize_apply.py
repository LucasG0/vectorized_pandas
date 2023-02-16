import inspect

import pytest

from apply import replace_apply


def compare(input_code: str, expected_code: str):
    input_code = inspect.cleandoc(input_code)
    expected_code = inspect.cleandoc(expected_code)

    result = replace_apply(input_code)
    assert result == expected_code, f"{result=} VS {expected_code=}"


class TestNoReplace:
    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            import json
            import pandas
            import numpy as np

            def func(val):
                a, b = test()
                a.x = c
                return val + a.x
            s = s.apply_func(func)
            print(s)
            """,
                """
            import json
            import pandas
            import numpy as np

            def func(val):
                a, b = test()
                a.x = c
                return val + a.x
            s = s.apply_func(func)
            print(s)
            """,
            )
        ],
    )
    def test_no_replace(self, input_code, expected_code):
        compare(input_code, expected_code)


class TestReplaceApplySimpleFunctions:
    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                return 1
            s = s.apply(func)
            """,
                """
            s = 1
            """,
            ),
            (
                """
            def func(val):
                return val + 1
            s = s.apply(func)
            """,
                """
            s = s + 1
            """,
            ),
        ],
    )
    def test_only_return(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                def func(val):
                    a = 1
                    return val + a
                s = s.apply(func)
                """,
                """
                s = s + 1
                """,
            ),
            (
                """
                def func(val):
                    a = 2
                    b = val
                    a += b
                    return a + a
                s = s.apply(func)
                """,
                """
                s = (2 + s) + (2 + s)
                """,
            ),
        ],
    )
    def test_intermediate_variables(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                import pandas as pd
                s = s.apply(pd.isna)
                """,
                """
                import pandas as pd
                s = pd.isna(s)
                """,
            ),
            (
                """
                import pandas as pd
                import numpy as np
                def func(val):
                    res = np.log(val) + 1
                    return np.isnan(pd.isna(res) * np.exp(2))
                s = s.apply(func)
                """,
                """
                import pandas as pd
                import numpy as np
                s = np.isnan(pd.isna(np.log(s) + 1) * np.exp(2))
                """,
            ),
        ],
    )
    def test_function_call(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                s = s.apply(lambda val: val + 1)
                """,
                """
                s = s + 1
                """,
            ),
        ],
    )
    def test_lambda_function(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                import pandas as pd
                s = s.apply(int)
                s = s.apply(np.float)
                """,
                """
                import pandas as pd
                s = s.astype(int)
                s = s.astype(np.float)
                """,
            ),
            (
                """
                def func(val):
                    return float(np.int(val))
                s = s.apply(func)
                """,
                """
                s = s.astype(np.int).astype(float)
                """,
            ),
            (
                """
                    s = s.apply(lambda x: str(x))
                    """,
                """
                    s = s.astype(str)
                    """,
            ),
        ],
    )
    def test_type_cast(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                    def func(val):
                        return val.capitalize()
                    s = s.apply(func)
                    """,
                """
                    s = s.str.capitalize()
                    """,
            ),
            (
                """
                def func(val):
                    return val.capitalize().startswith("A")
                s = s.apply(func)
                """,
                """
                s = s.str.capitalize().str.startswith("A")
                """,
            ),
            (
                """
                def func(val):
                    return ((val + "b").capitalize() + "a").startswith("A")
                s = s.apply(func)
                """,
                """
                s = ((s + "b").str.capitalize() + "a").str.startswith("A")
                """,
            ),
        ],
    )
    def test_str_methods(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                    def nested_func(x):
                        return x + 1

                    def func(val):
                        a = val + 2
                        return nested_func(a)

                    s = s.apply(func)
                """,
                """
                    def nested_func(x):
                        return x + 1


                    s = (s + 2) + 1
                """,
            ),
        ],
    )
    def test_nested_func(self, input_code, expected_code):
        compare(input_code, expected_code)


class TestReplaceApplyConditionalFunctions:
    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                if val == 1:
                    res = "A"
                else:
                    res = "C"
                return res
            s = s.apply(func)
            """,
                """
            import numpy as np
            s = np.select(conditions=[(s == 1)], choices=["A"], default="C")
            """,
            ),
            (
                """
            def func(val):
                if val == 1:
                    res = "A"
                elif val == 2:
                    res = "B"
                else:
                    res = "C"
                return res
            s = s.apply(func)
            """,
                """
            import numpy as np
            s = np.select(conditions=[(s == 1), (s == 2)], choices=["A", "B"], default="C")
            """,
            ),
        ],
    )
    def test_simple_conditions(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                res = "A"
                if val == 1:
                    res = "B"
                return res
            s = s.apply(func)
            """,
                """
            import numpy as np
            s = np.select(conditions=[(s == 1)], choices=["B"], default="A")
            """,
            ),
            (
                """
            def func(val):
                if val == 1:
                    res = "A"
                else:
                    res = "C"
                res = 1
                return res
            s = s.apply(func)
            """,
                """
            s = 1
            """,
            ),
        ],
    )
    def test_variable_defined_outside_conditions(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                if val == 1:
                    res = "B"
                    if val == 5:
                        res = "A"
                else:
                    if val == 10:
                        res = "C"
                    else:
                        res = "D"
                return res
            s = s.apply(func)
            """,
                """
            import numpy as np
            s = np.select(conditions=[(s == 1) & (s == 5), (s == 1) & ~((s == 5)), (s == 10)], choices=["A", "B", "C"], default="D")
            """,
            )
        ],
    )
    def test_multiple_simple_conditions(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                if val == 1:
                    return "B"
                elif val == 2:
                    return "C"
                return val
            s = s.apply(func)
            """,
                """
            import numpy as np
            s = np.select(conditions=[(s == 1), (s == 2)], choices=["B", "C"], default=s)
            """,
            )
        ],
    )
    def test_multiple_returns(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            import numpy as np
            def func(val):
                if val == 1:
                    res = "A"
                else:
                    res = "C"
                return res
            s = s.apply(func)
            """,
                """
            import numpy as np
            s = np.select(conditions=[(s == 1)], choices=["A"], default="C")
            """,
            ),
            (
                """
            import numpy
            def func(val):
                if val == 1:
                    res = "A"
                else:
                    res = "C"
                return res
            s = s.apply(func)
            """,
                """
            import numpy
            s = numpy.select(conditions=[(s == 1)], choices=["A"], default="C")
            """,
            ),
        ],
    )
    def test_numpy_already_imported(self, input_code, expected_code):
        compare(input_code, expected_code)


class TestReplaceApplyDataFrame:
    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                def func(col):
                    return col
                var_not_starting_with_df["col3"] = var_not_starting_with_df.apply(func, axis=0)
                """,
                """
                def func(col):
                    return col
                var_not_starting_with_df["col3"] = var_not_starting_with_df.apply(func, axis=0)
                """,
            ),
        ],
    )
    def test_no_replace_axis_0_explicit(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                def func(col):
                    return col
                df["col3"] = df.apply(func)
                """,
                """
                def func(col):
                    return col
                df["col3"] = df.apply(func)
                """,
            ),
        ],
    )
    def test_no_replace_axis_0_starts_with_df(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                def func(row):
                    a = row["col1"]
                    b = row[0] * 2  # unusual integer column name
                    return a + b
                df["col3"] = df.apply(func, axis=1)
                """,
                """
                df["col3"] = df["col1"] + (df[0] * 2)
                """,
            ),
            (
                """
            def func(row):
                if row["col1"] == 1:
                    res = "A"
                elif row["col2"] == 2:
                    res = "B"
                else:
                    res = "C"
                return res
            df["col3"] = df.apply(func, axis=1)
            """,
                """
            import numpy as np
            df["col3"] = np.select(conditions=[(df["col1"] == 1), (df["col2"] == 2)], choices=["A", "B"], default="C")
            """,
            ),
        ],
    )
    def test_df_apply(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                def func(val):
                    return val + 1
                df["col3"] = df["col1"].apply(func)
                """,
                """
                df["col3"] = df["col1"] + 1
                """,
            ),
        ],
    )
    def test_df_column_apply(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                def func(val):
                    return val + 1
                df.col3 = df.col1.apply(func)
                """,
                """
                df.col3 = df.col1 + 1
                """,
            ),
            (
                """
                def func(row):
                    return row.col1 + row["col2"]
                df.col3 = df.apply(func, axis=1)
                """,
                """
                df.col3 = df.col1 + df["col2"]
                """,
            ),
        ],
    )
    def test_df_dot_notation(self, input_code, expected_code):
        compare(input_code, expected_code)

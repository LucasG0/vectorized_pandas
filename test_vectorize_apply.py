import difflib
import inspect

import pytest

from apply import replace_apply


def compare(input_code: str, expected_code: str):
    input_code = inspect.cleandoc(input_code)
    expected_code = inspect.cleandoc(expected_code)

    result = replace_apply(input_code)
    if result != expected_code:
        print("{} => {}".format(result, expected_code))
        for i, s in enumerate(difflib.ndiff(result, expected_code)):
            if s[0] == " ":
                continue
            elif s[0] == "-":
                print('Delete "{}" from position {}'.format(s[-1], i))
            elif s[0] == "+":
                print('Add "{}" to position {}'.format(s[-1], i))
        print()
        raise ValueError(f"\n{result=}\n{expected_code=}")


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
                (a, b) = test()
                a.x = c
                return val + a.x


            s = s.apply_func(func)
            print(s)
            """,
            ),
            # (
            #     """
            #     import json
            #
            #     def func(row):
            #         return json.loads(row)
            #     x = df.apply(func, axis=1)
            #     """,
            #     """
            #     import json
            #
            #
            #     def func(row):
            #         return json.loads(row)
            #
            #
            #     x = df.apply(func, axis=1)
            #     """,
            # ),
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
                s = 2 + s + (2 + s)
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


                    s = s + 2 + 1
                """,
            ),
        ],
    )
    def test_nested_func(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                b = "b"
                return f"{val}_{b}"
            s = s.apply(func)
            """,
                """
            s = s.astype(str) + "_" + "b"
            """,
            ),
            (
                """
            def func(val):
                a = 1
                return f"_{val + 2}_{a}_"
            s = s.apply(func)
            """,
                """
            s = "_" + (s + 2).astype(str) + "_" + str(1) + "_"
            """,
            ),
        ],
    )
    def test_f_strings(self, input_code, expected_code):
        compare(input_code, expected_code)


class TestReplaceApplyAnyLocation:
    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                return val + 1

            def main():
                s = s.apply(func)
                return s
            """,
                """
            def main():
                s = s + 1
                return s
            """,
            ),
            (
                """
                def func(val):
                    return val + 1

                def main():
                    return s.apply(func)
                """,
                """
                def main():
                    return s + 1
                """,
            ),
        ],
    )
    def test_apply_call_within_function(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                return val + 1

            class A:
                def method(self):
                    s = s.apply(func)

            def main():
                a = A()
                a.method()
            """,
                """
            class A:
                def method(self):
                    s = s + 1


            def main():
                a = A()
                a.method()
            """,
            ),
        ],
    )
    def test_apply_call_within_method(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                return val + 2 * val

            val = s.apply(func).sum()
            """,
                """
            val = (s + 2 * s).sum()
            """,
            ),
        ],
    )
    def test_dummy_chained_assignment(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                return val + 2 * val

            s = s.isna().apply(func)
            """,
                """
            s_temp = s.isna()
            s = s_temp + 2 * s_temp
            """,
            ),
        ],
    )
    def test_chained_assignment(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                return val + 2 * val

            if True:
                s = 1
            else:
                s = s.isna().apply(func)
            """,
                """
            if True:
                s = 1
            else:
                s_temp = s.isna()
                s = s_temp + 2 * s_temp
            """,
            ),
        ],
    )
    def test_orelse_chained_assignment(self, input_code, expected_code):
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

            s = np.select(conditions=[s == 1], choices=["A"], default="C")
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

            s = np.select(conditions=[s == 1, s == 2], choices=["A", "B"], default="C")
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

            s = np.select(conditions=[s == 1], choices=["B"], default="A")
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

            s = np.select(conditions=[(s == 1) & (s == 5), s == 1, s == 10], choices=["A", "B", "C"], default="D")
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

            s = np.select(conditions=[s == 1, s == 2], choices=["B", "C"], default=s)
            """,
            ),
            (
                """
                def func(val):
                    if val == 1:
                        return "B"
                    elif val == 2:
                        return "C"
                    else:
                        return val

                s = s.apply(func)
                """,
                """
                import numpy as np

                s = np.select(conditions=[s == 1, s == 2], choices=["B", "C"], default=s)
                """,
            ),
            (
                """
                def func(val):
                    if val == 1:
                        return "B"
                    if val == 2:
                        return "C"
                    return val

                s = s.apply(func)
                """,
                """
                import numpy as np

                s = np.select(conditions=[s == 1, s == 2], choices=["B", "C"], default=s)
                """,
            ),
            (
                """
                def func(row):
                    if row["A"] == 1:
                        if row["B"] == 1:
                            return row["C"]
                        else:
                            return row["D"]
                    else:
                        if row["B"] == 2:
                            return row["E"]
                        else:
                            return 0

                df["H"] = df.apply(func, axis=1)
                """,
                """
                import numpy as np

                df["H"] = np.select(
                    conditions=[(df["A"] == 1) & (df["B"] == 1), df["A"] == 1, df["B"] == 2],
                    choices=[df["C"], df["D"], df["E"]],
                    default=0,
                )
                """,
            ),
            (
                """
                def func(row):
                    if row["A"] != 0:
                        if row["A"] == 1:
                            if row["B"] == 2:
                                return row["C"]
                            else:
                                return row["D"]
                        else:
                            if row["B"] == 3:
                                return row["E"]
                            else:
                                return row["F"]
                    else:
                        return 0.0

                df["H"] = df.apply(func, axis=1)
                """,
                """
                import numpy as np

                df["H"] = np.select(
                    conditions=[
                        (df["A"] != 0) & (df["A"] == 1) & (df["B"] == 2),
                        (df["A"] != 0) & (df["A"] == 1),
                        (df["A"] != 0) & (df["B"] == 3),
                        df["A"] != 0,
                    ],
                    choices=[df["C"], df["D"], df["E"], df["F"]],
                    default=0.0,
                )
                """,
            ),
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

            s = np.select(conditions=[s == 1], choices=["A"], default="C")
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

            s = numpy.select(conditions=[s == 1], choices=["A"], default="C")
            """,
            ),
        ],
    )
    def test_numpy_already_imported(self, input_code, expected_code):
        compare(input_code, expected_code)

    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
            def func(val):
                return val if val == 0 else 0

            s = s.apply(func)
            """,
                """
            import numpy as np

            s = np.select(conditions=[s == 0], choices=[s], default=0)
            """,
            ),
            (
                """
            def func(row):
                a = "A" if row["col1"] == 0 else "B"
                if row["col2"] == 1:
                    return a
                return "C"

            s = df.apply(func, axis=1)
            """,
                """
            import numpy as np

            s = np.select(conditions=[(df["col2"] == 1) & (df["col1"] == 0), df["col2"] == 1], choices=["A", "B"], default="C")
            """,
            ),
        ],
    )
    def test_if_expression(self, input_code, expected_code):
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
                df["col3"] = df["col1"] + df[0] * 2
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

            df["col3"] = np.select(conditions=[df["col1"] == 1, df["col2"] == 2], choices=["A", "B"], default="C")
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


class TestGlobalVariables:
    @pytest.mark.parametrize(
        "input_code,expected_code",
        [
            (
                """
                A = 1
                B = 2

                def func(val):
                    if val == B:
                        return val + A
                    return val

                s = s.apply(func)
                """,
                """
                import numpy as np

                A = 1
                B = 2
                s = np.select(conditions=[s == B], choices=[s + A], default=s)
                """,
            ),
        ],
    )
    def test_global_variables(self, input_code, expected_code):
        compare(input_code, expected_code)

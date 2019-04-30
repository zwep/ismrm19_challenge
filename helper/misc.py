# encoding: utf-8

"""
Some misc helpers
"""

import re
import warnings
import math


def overrides(interface_class):
    """
    Used to in parent/child methods to prevent stuff being overwritten. See StackOverflow link

    https://stackoverflow.com/questions/1167617/in-python-how-do-i-indicate-im-overriding-a-method
    :param interface_class:
    :return:
    """

    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


def simple_div(n):
    """
    Calculate the divisors of n. Used to split arrays into a groups of integer size
    :param n:
    :return:
    """
    return [i for i in range(n, 0, -1) if n % i == 0]


def get_square(x):
    """
    Used to get an approximation of the square of a number.
    Needed to place N plots on a equal sized grid.
    :param x:
    :return:
    """
    x_div = simple_div(x)
    x_sqrt = math.sqrt(x)
    diff_list = [abs(y - x_sqrt) for y in x_div]
    res = diff_list.index(min(diff_list))
    return x_div[res], x // x_div[res]


def type2list(x):
    """
    Converts almost anything to a list.
    Needed when building a configuration dictionary for model runs. To be sure that we can iterate over the object
    :param x:
    :return:
    """
    if isinstance(x, list):
        return x
    elif isinstance(x, int):
        return [x]
    elif isinstance(x, str):
        return [x]
    elif isinstance(x, float):
        return [x]
    else:
        warnings.warn('type2list escaped on value: {}'.format(str(x)))


def filter_tuple_list(input_key, param_list):
    """
    Used to filter a list of tuples that have the structure [("key", value), ...]
    # TODO Waarom gebruik je niet gewoon een dict hiervoor...????

    example
    A = dict(zip(list(string.ascii_letters), range(26*2)))
    B = list(map(tuple, zip(list(string.ascii_letters), range(26*2))))

    :param input_key:
    :param param_list:
    :return:
    """
    param_res = [x[1] for x in param_list if re.match("^" + input_key + "$", x[0])]
    if len(param_res) == 1:
        param_sel = param_res[0]
    else:
        param_sel = None
    return param_sel


def dictlist2listdict(dict_list):
    # Transforms a dictionary of lists to a list of dictionaries
    return [dict(zip(dict_list, t)) for t in zip(*dict_list.values())]


def get_nested(dict, keys):
    # TODO Usage of class names in function..?
    res = dict
    for key in keys:
        res = res.get(key)
    return res


def set_nested(dic, keys, value):
    """

    :param dic:
    :param keys:
    :param value:
    :return:
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

# Useful to test functionality of functions
# from timeit import Timer
# def foo():
#     return [np.random.randint(10) for i in range(18)]
#
# def foobar():
#     return np.random.randint(10, size=(18,))
#
# t1 = Timer("""foo()""", """from __main__ import foo""")
# t2 = Timer("""foobar()""", """from __main__ import foobar""")
# t1.timeit(5000)  # runs foo() 1000 times and returns the time taken
# t2.timeit(5000)
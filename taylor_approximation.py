"""MODULE TO WORK WITH TAYLOR SERIES APPROXIMATION
Draw different dependence graphs and find value of the function, such as:
based on the sum of severe first elements in point x.

Function: approximated_result_for_first_num_elements(number_of_elements: int, x_variable: float)


Function names to get data to draw the plot

:number_epsilon_dependency_for_stated_x
:number_x_var_dependency_for_stated_epsilon
:difference_x_var_dependency_for_stated_number
:difference_index_element_dependency_for_stated_x

Draw the plot with using functions:
:draw_graph
:compare_graphs
"""

from math import sin, log10 as lg
from functools import cache
from sys import setrecursionlimit
from matplotlib import pyplot as plt
import numpy as np
setrecursionlimit(30000)

# -------------------------------------------------------------------------------------------------
# ---------------------------------------FUNCTION APPROXIMATION PART--------------------------------
# -------------------------------------------------------------------------------------------------


@cache
def get_row_value(x_variable: float, k: int) -> float:
    """
    Calculate and return value of the element k in Taylor
     series (for a=0)
    Parameters:
    :k - integer number of current serial element to find
    :x_variable - point in which calculate a serial element
    >>> get_row_value(1, 1)
    64.0
    >>> get_row_value(0.1,75)
    2.620823677304307e-254
    >>> get_row_value(0.5,10)
    -0.00010734266046787416
    """
    return -3/4 * (-1)**(k % 2) * pow_cached(4, 1+2*k) * (-1 + pow_cached(9, k))\
                * pow_cached(x_variable, 2*k+1)/factorial_cached(1+2*k) if k < 75 else\
        (-1) * 16 * 9 * (x_variable**2)/(1+2*k) / \
        (2*k) * get_row_value(x_variable, k-1)


@cache
def factorial_cached(number: int) -> int:
    """Find and return the factorial of the number. Cached to work faster.
    >>> factorial_cached(0)
    1
    >>> factorial_cached(10)
    3628800
    >>> factorial_cached(-1)
    """
    return None if number < 0 else 1 if number < 2 else number * factorial_cached(number-1)


@cache
def pow_cached(number: float, exp: int) -> float:
    """Find and return value of the number in exponent - exp. Cached to work faster.
    :param number: - float number to be raised to the stated degree
    :param exp: - int number degree to which raise the number
    >>> pow_cached(3, 7)
    2187.0
    >>> pow_cached(2, 101)
    2535301200456458802993406410752
    >>> pow_cached(1001, -1)
    """
    return None if exp < 0 else 1 if exp == 0 else number * pow_cached(number, exp-1)


def built_in_result(x_variable: float) -> float:
    """Return value of the function in stated valiable x
    of the built-in python function of function sin(4x)**3
    >>> built_in_result(0)
    0.0
    >>> built_in_result(1)
    -0.4334586419808374
    """
    return sin(4*x_variable)**3


# -------------------------------------------------------------------------------------------------
# ---------------------------------DIFFERENCE TO BUILT-IN PART--------------------------------------
# -------------------------------------------------------------------------------------------------

@cache
def aproximated_result_for_first_num_elements(number_of_elements: int, x_variable: float) -> float:
    """ Calculate and return function approximated result for the function
    as the sum of several first terms in Taylor
     series.
    Parameters:
    :number_of_elements - number of first series elements to consider
    :x_variable - point in which find approximated result
    """
    return None if number_of_elements < 1 else get_row_value(x_variable, number_of_elements)\
        if number_of_elements == 1 else get_row_value(x_variable, number_of_elements)\
        + aproximated_result_for_first_num_elements(number_of_elements-1, x_variable)


def get_number_of_elements(epsilon: float, x_variable: float) -> int:
    """ Return number of first series element needed to make difference
    of approximated result less than epsilon
    Parameters:
    :epsilon - check difference between result and built-in function
    :x_variable point at which evaluate function approximation result
    """
    RIGHT_ANSWER, series_ind = built_in_result(x_variable), 0
    while True:
        series_ind += 1
        current_result = aproximated_result_for_first_num_elements(
            series_ind, x_variable)
        if abs(current_result - RIGHT_ANSWER) < epsilon:
            return series_ind

# -------------------------------------------------------------------------------------------------
# ---------------------------------------GRAPHS PART-----------------------------------------------
# -------------------------------------------------------------------------------------------------


def draw_graph(x_vars: list, Y_DATA: dict[float | int:list], x_label: str, y_label: str,
               plot_title: str, show=True, save=False) -> None:
    """  MAIN FUNCTION TO DRAW GRAPHS
    It is universalized for using with different graph formats from that module.
    Have two optional parameters, show and save
    Parameters:
    :x_vars - sequential x-coordinates of data points in list format
    :Y_DATA - dictionary with y-coordinates data of the points for
    different key functions from their keys that are labels for the lines,
    values of that dictionary are lists with sequential y_coords of the points
    :x_label - label for the x-axis name
    :y_label - label for the y-axis name
    :plot_title - title for the graph
    :show -bool optional parameter. If True then show the graph for user.
    If it is False do not show.  Default :param show: value True.
    :save -bool optional parameter. If True then save the graph for user.
    If it is False do not save.  Default :param show: value False.
    """
    COLOURS = {'red', 'black', 'yellow', 'green', 'blue'}
    plt.grid(color='grey', linestyle='dashdot', linewidth=0.2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    for lable, y_data in Y_DATA.items():
        plt.plot(x_vars, y_data, label=lable, color=COLOURS.pop())
        if not COLOURS:
            COLOURS = {'red', 'black', 'yellow', 'green', 'blue'}
    plt.legend()
    if show:
        plt.show()
    if save:
        plt.savefig("graph")


def number_epsilon_dependancy_for_stated_x(x_var_list: list[float], epsylon_list: list[float])\
        -> tuple[list[float], dict[float:list], str, str, str]:
    """
    Function to find and return data for the plot of (epsilon - number of elements) dependance,
    for different starting x variables.
    Where epsilon is the neighbourhood from the built-in function result.
    Number of elements - required minimal number of first terms from Taylor
     series that difference
    from the aproximation result to the built-in functiion value was less then epsilon.
    Shortly, for each point x from x_var_list we find dependency of
    epsylon number to the minimum number of Taylor
     series elements.
    Parameters:
    :x_var_list - list with x-coords of point to calculate function result in them
    :epsylon_list - list with epsilon values to take into account. Epsylon is a float number
    to check if result of the function aproximation is less then its built-in result
    Return:
    :arg1 -list with float numbers that are equel to -lg(epsylon) for epsylon values from
    epsylon_list. They will be used as x-coordinates for the graph.
    :arg2 -  dictionary with y-coordinates data of the points for
    different key functions main arguments f(x, z). Keys are labels for the lines
    and are equel to x_var_list values representing epsylon values in logarithmic format.
    Values of that dictionary are lists with sequential y_coords of the points.
    They represente minimum number of the series elements reqired to make difference from the
    built-in function and aproximation result less then epsilon, that are in dictionary keys.
    :arg3 - x-artix label for the plot
    :arg4 - y-artix label for the plot
    :arg5 - title name for the graph
    """
    plot_x_values = [-lg(eps) for eps in epsylon_list]
    Y_DATA = {x: [get_number_of_elements(eps, x) for eps in epsylon_list] for x in x_var_list}
    return plot_x_values, Y_DATA, '- lg(epsilon)', 'number of elements',\
           'Dependency of minimum elements number from epsilon for different x'


def number_x_var_dependency_for_stated_epsilon(x_var_list:list[float], epsilon_list:list[float])\
                           -> tuple[list[float], dict[float:list], str, str, str]:
    """
    Function to find and return data for the plot of (x_var - number of elements) dependence,
    for different epsilon values.
    Where epsilon is the neighbourhood from the built-in function result.
    Number of elements - required minimal number of first terms from Taylor
     series that difference
    from the approximation result to the built-in function value was less than epsilon.
    Shortly, for each epsilon value from the epsilon_list we find dependency
    of x to the minimum number of Taylor
     series elements.
    Parameters:
    :x_var_list - list with x-coords of point to calculate function result in them
    :epsilon_list - list with epsilon values to take into account. Epsilon is a float number
    to check if result of the function approximation is less than its built-in result
    Return:
    :arg1 -list with x-coordinates numbers from x_var_list for the graph.
    :arg2 -  dictionary with y-coordinates data of the points for
    different key functions main arguments f(x, z). Keys are labels for the lines
    and are equal to epsilon values representing the neighbourhood from the built-in function
    result and approximation from Taylor
     series. Values of that dictionary are lists with minimum
    number of first elements required to make difference less than epsilon, which also are their
    keys, they will be used as y_cords of the graph points for line with label key=epsilon.
    arg3 - x-artis label for the plot
    :arg4 - y-artis label for the plot
    :arg5 - title name for the graph
    """
    Y_DATA = {eps: [get_number_of_elements(eps, x) for x in x_var_list] for eps in epsilon_list}
    return x_var_list, Y_DATA, 'x', 'number of elements',\
            'Dependency of minimum elements number from x for different epsilon'


def difference_index_element_dependency_for_stated_x(x_var_list:list[float], number_list: list[int])\
                           -> tuple[list[float], dict[float:list], str, str, str]:
    """
    Function to find and return data for the plot of (difference - number of elements) dependence,
    for different x values.
    Difference - the difference between severe first terms from Taylor
     series and built-in function
    result.
    Number of elements - required minimal number of first terms from Taylor
     series that difference
    from the approximation result to the built-in function value was less than epsilon.
    Shortly, for each x value from the x_var_list we find dependency
    of function difference to the minimum number of Taylor
     series elements.
    Parameters:
    :x_var_list - list with x-cords of point to calculate function result in them
    :number_list - list with numbers of how many first elements from the Taylor series
    to take into account.
    Return:
    :arg1 - number_list with num of elements to take from Taylor series.
    They will be taken as x_coordinated for the graph.
    :arg2 -  dictionary with y-coordinates data of the points for
    different key functions main arguments f(x, z). Keys are labels for the lines
    and are equal to x_var. Values of that dictionary are
    They will be used as y_cords of the graph points for line with label key=epsilon.
    arg3 - x-artis label for the plot
    :arg4 - y-artis label for the plot
    :arg5 - title name for the graph
    """
    Y_DATA = {x: [] for x in x_var_list}
    for num in number_list:
        for x in x_var_list:
            value = aproximated_result_for_first_num_elements(
                num, x) - built_in_result(x)
            Y_DATA[x].append(0 if not value else lg(abs(value)))
    return number_list, Y_DATA, 'number of elements', 'lg(difference)',\
           'Difference to built-in function from series with limited elements number, for different x'


def difference_x_var_dependency_for_stated_number(x_var_list:list[float], number_list: list[int])\
                               -> tuple[list[float], dict[float:list], str, str, str]:
    """
    Function to find and return data for the plot of (difference - x values) dependence,
    for different number of first Taylor series elements.
    Difference - the difference between severe first terms from tear series and built-in function
    result.
    Number of elements - required minimal number of first terms from tear series that difference
    from the approximation result to the built-in function value was less than epsilon.
    Shortly, for each number of first elements from number_list find dependency
    of function difference to x variable.
    Parameters:
    :x_var_list - list with x-cords of point to calculate function result in them
    :number_list - list with numbers of how many first elements from the Taylor series
    to take into account.
    Return:
    :arg1 -list with x-coordinates numbers from x_var_list for the graph.
    :arg2 -  dictionary with y-coordinates data of the points for
    different key functions main arguments f(x, z). Keys are labels for the lines
    and are equal to x_var. Values are logarithmic value of difference between severe
    first terms from tear series and built-in function result.
    They will be used as y_cords of the graph points for line with
    label key=number of first elements to take.
    :arg3 - x-artis label for the plot
    :arg4 - y-artis label for the plot
    :arg5 - title name for the graph
    """
    Y_DATA = {num: [] for num in number_list}
    for x in x_var_list:
        for num in number_list:
            value = aproximated_result_for_first_num_elements(
                num, x) - built_in_result(x)
            Y_DATA[num].append(0 if not value else lg(abs(value)))
    return x_var_list, Y_DATA, 'x', 'lg(difference)',\
           'Difference to built-in function from x, for different number of series elements'


def compare_graphs(x_vars:list[float], number_list:list[int], show=True, save=False) -> None:
    """
    Draw, show and save on request the graph of built-in function.
    And approximation from Taylor series for severe(number from number_list) elements.
    Parameters:
    :x_var_list - list with x-cords of point to calculate function result in them
    :number_list - list with numbers of how many first elements from the Taylor series
    to take into account.
    show -bool optional parameter. If True then shows the graph for user.
    If it is False do not show.  Default :param show: value True.
    save -bool optional parameter. If True then saves the graph for user.
    If it is False do not save.  Default :param show: value False.
    """
    COLOURS = ['red', 'black', 'yellow', 'green']
    plt.grid(color='grey', linestyle='dashdot', linewidth=0.2)
    plt.title('Comparing built-in function graph to our\
approximations for different numbers of first elements')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_vars, np.sin(4*x_vars)**3, label='built-in', color='blue')
    for num in number_list:
        plt.plot(x_vars, [aproximated_result_for_first_num_elements(num, x)
                 for x in x_vars], label=str(num), color=COLOURS.pop())
    plt.legend()
    if show:
        plt.show()
    if save:
        plt.savefig("graph")



# -------------------------------------------------------------------------------------------------
# -----------------------------------USER INTERFACE PART-------------------------------------------
# -------------------------------------------------------------------------------------------------


def get_input():
    """  GET USER INPUT. Show or save the graph draw on input data  """
    try:
        min_num = float(input('Print minimum value for x on graph\n>>> '))
        max_num = float(input('Print minimum value for x on graph\n>>> '))
        step = float(input('Print step value for x on graph\n>>> '))
        print('Print in different lines(separated with ENTER) number of elements of the Taylor\
               series to take into account. If you want stop print empty line(press ENTER)')
        numbers = []
        while True:
            input_text = input('>>> ')
            if input_text:
                numbers.append(int('>>> '))
            else:
                break
        show = bool(input('Do you want to see the graph? If yes print anything'))
        save = bool(input('Do you want to save the graph in current directory? If yes print anything'))
        compare_graphs(np.arange(min_num, max_num, step), numbers, show, save)
    except (ValueError, TypeError, IndexError):
        print('Oops.. Something went wrong. If you want run the program again.')

if __name__ == '__main__':
    get_input()

from math import factorial, gcd, sqrt
from statistics import mean, median
from typing import List

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arith")


# ----------------------------------------------------
# Basic Arithmetic
# ----------------------------------------------------

@mcp.tool()
def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Return the difference of two numbers."""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@mcp.tool()
def modulus(a: float, b: float) -> float:
    """Return remainder after dividing a by b."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a % b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise a number to a power."""
    return base ** exponent


# ----------------------------------------------------
# Number Theory
# ----------------------------------------------------

@mcp.tool()
def factorial_number(n: int) -> int:
    """Return factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is defined only for non-negative integers.")
    return factorial(n)


@mcp.tool()
def greatest_common_divisor(a: int, b: int) -> int:
    """Return greatest common divisor."""
    return gcd(a, b)


@mcp.tool()
def least_common_multiple(a: int, b: int) -> int:
    """Return least common multiple."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


@mcp.tool()
def is_prime(n: int) -> bool:
    """Check whether a number is prime."""
    if n < 2:
        return False

    limit = int(sqrt(n))

    for i in range(2, limit + 1):
        if n % i == 0:
            return False

    return True


@mcp.tool()
def is_even(n: int) -> bool:
    """Return True if number is even."""
    return n % 2 == 0


@mcp.tool()
def is_odd(n: int) -> bool:
    """Return True if number is odd."""
    return n % 2 != 0


# ----------------------------------------------------
# Statistics
# ----------------------------------------------------

@mcp.tool()
def average(numbers: List[float]) -> float:
    """Return arithmetic mean."""
    if not numbers:
        raise ValueError("List cannot be empty.")
    return mean(numbers)


@mcp.tool()
def median_value(numbers: List[float]) -> float:
    """Return median."""
    if not numbers:
        raise ValueError("List cannot be empty.")
    return median(numbers)


@mcp.tool()
def maximum(numbers: List[float]) -> float:
    """Return largest number."""
    if not numbers:
        raise ValueError("List cannot be empty.")
    return max(numbers)


@mcp.tool()
def minimum(numbers: List[float]) -> float:
    """Return smallest number."""
    if not numbers:
        raise ValueError("List cannot be empty.")
    return min(numbers)


@mcp.tool()
def sum_numbers(numbers: List[float]) -> float:
    """Return sum of a list."""
    return sum(numbers)


# ----------------------------------------------------
# Comparison
# ----------------------------------------------------

@mcp.tool()
def maximum_of_two(a: float, b: float) -> float:
    """Return larger of two numbers."""
    return max(a, b)


@mcp.tool()
def minimum_of_two(a: float, b: float) -> float:
    """Return smaller of two numbers."""
    return min(a, b)


@mcp.tool()
def compare_numbers(a: float, b: float) -> str:
    """Compare two numbers."""
    if a > b:
        return f"{a} is greater than {b}"
    if a < b:
        return f"{a} is less than {b}"
    return "Both numbers are equal."


# ----------------------------------------------------
# Utility
# ----------------------------------------------------

@mcp.tool()
def absolute_value(x: float) -> float:
    """Return absolute value."""
    return abs(x)


@mcp.tool()
def square_root(x: float) -> float:
    """Return square root."""
    if x < 0:
        raise ValueError("Square root of a negative number is not supported.")
    return sqrt(x)


@mcp.tool()
def percentage(part: float, whole: float) -> float:
    """Calculate what percentage part is of whole."""
    if whole == 0:
        raise ValueError("Whole cannot be zero.")
    return (part / whole) * 100


@mcp.tool()
def percentage_increase(old: float, new: float) -> float:
    """Return percentage increase."""
    if old == 0:
        raise ValueError("Old value cannot be zero.")
    return ((new - old) / old) * 100


@mcp.tool()
def percentage_decrease(old: float, new: float) -> float:
    """Return percentage decrease."""
    if old == 0:
        raise ValueError("Old value cannot be zero.")
    return ((old - new) / old) * 100


if __name__ == "__main__":
    mcp.run(transport="stdio")
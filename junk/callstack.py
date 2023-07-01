def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


out = factorial(5)
print(out)


f = lambda x: 2 * x

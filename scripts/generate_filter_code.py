from sympy import *

# Generate C++ code for the Cloude's Mueller matrix filtering.

# Python implementation is based on the following.
# https://github.com/LogikerKit/MuellerConeFilter


T = 0.25 * Matrix(
    [
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, -I, 0, 0, I, 0],
        [0, 0, 1, 0, 0, 0, 0, I, 1, 0, 0, 0, 0, -I, 0, 0],
        [0, 0, 0, 1, 0, 0, -I, 0, 0, I, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, I, 0, 0, -I, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
        [0, 0, 0, I, 0, 0, 1, 0, 0, 1, 0, 0, -I, 0, 0, 0],
        [0, 0, I, 0, 0, 0, 0, 1, -I, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, -I, 1, 0, 0, 0, 0, I, 0, 0],
        [0, 0, 0, -I, 0, 0, 1, 0, 0, 1, 0, 0, I, 0, 0, 0],
        [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
        [0, I, 0, 0, -I, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, I, 0, 0, -I, 0, 0, 1, 0, 0, 0],
        [0, 0, -I, 0, 0, 0, 0, 1, I, 0, 0, 0, 0, 1, 0, 0],
        [0, -I, 0, 0, I, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1],
    ]
)  # first row  # second row  # third row  # fourth row
Tinv = T.inv()


def main():
    h = symbols("h0:16", real=True)
    h = Matrix(16, 1, h)

    m = symbols("m0:16", real=True)
    m = Matrix(16, 1, m)

    Tm = T * m
    Tm4x4 = Matrix(4, 4, Tm)
    for j in range(4):
        for i in range(4):
            eq = Tm4x4[i, j]
            eq = factor(eq)

            eq_real = str(factor(simplify(re(eq))))
            eq_imag = str(factor(simplify(im(eq))))

            # m0 -> m(0)
            # m1 -> m(1)
            # ...
            for k in reversed(range(16)):
                eq_real = eq_real.replace(f"m{k}", f"m({k})")
                eq_imag = eq_imag.replace(f"m{k}", f"m({k})")

            print(f"H({i}, {j}) = std::complex<float>({eq_real}, {eq_imag});")

    print()

    m_ = Tinv * h
    for i in range(16):
        eq = m_[i]
        eq = simplify(eq)
        eq = str(eq)
        # print(eq)

        # h0 -> h(0)
        # h1 -> h(1)
        # ...
        for k in reversed(range(16)):
            eq = eq.replace(f"h{k}", f"h_({k})")

        eq = eq.replace("1.0*", "")

        print(f"m_({i}) = ({eq}).real();")


if __name__ == "__main__":
    main()

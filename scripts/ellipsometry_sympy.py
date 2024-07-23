from sympy import *
import polanalyser.sympy as pas


def main():
    theta = Symbol("theta", real=True)

    M = pas.mueller()
    print(M)

    M_obs = pas.polarizer(0) @ pas.qwp(5 * theta) @ M @ pas.qwp(theta) @ pas.polarizer(0)
    M_obs_00 = M_obs[0, 0]
    # print(M_obs_00)

    # for j in range(4):
    #     for i in range(4):
    #         if M[j, i] not in M_obs_00.free_symbols:
    #             coff = 0
    #         else:
    #             coff = Poly(M_obs_00, M[j, i]).coeffs()[0]

    #         print(f"m{j}{i}: {coff}")

    # exit()

    # log_M_obs_00 = ln(M_obs_00)

    # # Derivative of log_M_obs_00
    # dlog_M_obs_00 = diff(log_M_obs_00, theta)

    # final = factor(expand(simplify(dlog_M_obs_00)))
    # print(final)
    # print("--------------------")
    # # print(latex(final))

    # # Separate by numerator and denominator
    # num, den = fraction(final)

    m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33 = M

    num = (
        -8.0 * m01 * sin(2 * theta) * cos(2 * theta)
        - 4.0 * m02 * sin(2 * theta) ** 2
        + 4.0 * m02 * cos(2 * theta) ** 2
        + 4.0 * m03 * cos(2 * theta)
        - 20.0 * m10 * sin(20 * theta)
        - 8.0 * m11 * sin(2 * theta) * cos(2 * theta) * cos(10 * theta) ** 2
        - 20.0 * m11 * sin(20 * theta) * cos(2 * theta) ** 2
        - 4.0 * m12 * sin(2 * theta) ** 2 * cos(10 * theta) ** 2
        - 20.0 * m12 * sin(2 * theta) * sin(20 * theta) * cos(2 * theta)
        + 4.0 * m12 * cos(2 * theta) ** 2 * cos(10 * theta) ** 2
        - 20.0 * m13 * sin(2 * theta) * sin(20 * theta)
        + 4.0 * m13 * cos(2 * theta) * cos(10 * theta) ** 2
        - 20.0 * m20 * sin(10 * theta) ** 2
        + 20.0 * m20 * cos(10 * theta) ** 2
        - 4.0 * m21 * sin(2 * theta) * sin(20 * theta) * cos(2 * theta)
        - 20.0 * m21 * sin(10 * theta) ** 2 * cos(2 * theta) ** 2
        + 20.0 * m21 * cos(2 * theta) ** 2 * cos(10 * theta) ** 2
        - 2.0 * m22 * sin(2 * theta) ** 2 * sin(20 * theta)
        - 20.0 * m22 * sin(2 * theta) * sin(10 * theta) ** 2 * cos(2 * theta)
        + 20.0 * m22 * sin(2 * theta) * cos(2 * theta) * cos(10 * theta) ** 2
        + 2.0 * m22 * sin(20 * theta) * cos(2 * theta) ** 2
        - 20.0 * m23 * sin(2 * theta) * sin(10 * theta) ** 2
        + 20.0 * m23 * sin(2 * theta) * cos(10 * theta) ** 2
        + 2.0 * m23 * sin(20 * theta) * cos(2 * theta)
        - 20.0 * m30 * cos(10 * theta)
        + 8.0 * m31 * sin(2 * theta) * sin(10 * theta) * cos(2 * theta)
        - 20.0 * m31 * cos(2 * theta) ** 2 * cos(10 * theta)
        + 4.0 * m32 * sin(2 * theta) ** 2 * sin(10 * theta)
        - 20.0 * m32 * sin(2 * theta) * cos(2 * theta) * cos(10 * theta)
        - 4.0 * m32 * sin(10 * theta) * cos(2 * theta) ** 2
        - 20.0 * m33 * sin(2 * theta) * cos(10 * theta)
        - 4.0 * m33 * sin(10 * theta) * cos(2 * theta)
    )

    den = 2 * m00 + 2 * m01 * cos(2 * theta) ** 2 + 2 * m02 * sin(2 * theta) * cos(2 * theta) + 2 * m03 * sin(2 * theta) + 2 * m10 * cos(10 * theta) ** 2 + 2 * m11 * cos(2 * theta) ** 2 * cos(10 * theta) ** 2 + 2 * m12 * sin(2 * theta) * cos(2 * theta) * cos(10 * theta) ** 2 + 2 * m13 * sin(2 * theta) * cos(10 * theta) ** 2 + m20 * sin(20 * theta) + m21 * sin(20 * theta) * cos(2 * theta) ** 2 + m22 * sin(2 * theta) * sin(20 * theta) * cos(2 * theta) + m23 * sin(2 * theta) * sin(20 * theta) - 2 * m30 * sin(10 * theta) - 2 * m31 * sin(10 * theta) * cos(2 * theta) ** 2 - 2 * m32 * sin(2 * theta) * sin(10 * theta) * cos(2 * theta) - 2 * m33 * sin(2 * theta) * sin(10 * theta)

    print(num)
    print("--------------------")
    print(den)
    print("--------------------")

    # Get cofficients
    print("numerator")
    for j in range(4):
        for i in range(4):
            if M[j, i] not in num.free_symbols:
                coff = 0
            else:
                coff = Poly(num, M[j, i]).coeffs()[0]

            if i == 0 and j == 0:
                print("[", end="")

            print(f"{coff}", end="")

            if i == 3 and j == 3:
                print("]")
            else:
                print(",", end="")

            # print(f"m{j}{i}: {coff}")

    print("denominator")
    for j in range(4):
        for i in range(4):
            if M[j, i] not in den.free_symbols:
                coff = 0
            else:
                coff = Poly(den, M[j, i]).coeffs()[0]

            if i == 0 and j == 0:
                print("[", end="")
            print(f"{coff}", end="")
            if i == 3 and j == 3:
                print("]")
            else:
                print(",", end="")

            # print(f"m{j}{i}: {coff}")


if __name__ == "__main__":
    main()

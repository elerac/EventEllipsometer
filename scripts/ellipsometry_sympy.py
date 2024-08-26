from sympy import *
from sympy.codegen.rewriting import create_expand_pow_optimization
from sympy.simplify.cse_main import cse
import polanalyser.sympy as pas
import black


def decompose_coefficients(expr: Basic, M: Matrix = pas.mueller()) -> list:
    """Decompose coefficients of expression by Mueller matrix"""
    coffs = []
    for j in range(4):
        for i in range(4):
            if M[j, i] not in expr.free_symbols:
                coff = S(0.0)
            else:
                coff = Poly(expr, M[j, i]).coeffs()[0]

            coffs.append(coff)

    return coffs


def extract_unique_symbols(expr: Expr) -> dict:
    # Get all possible terms
    possible_terms = []
    for term_add in expr.as_ordered_terms():
        if not term_add.is_Mul:
            term = term_add
            # is it has ** remove it
            if term.is_Pow:
                term = term.base

            possible_terms.append(term)
            continue
        else:
            # Decompose by * operator
            for term in term_add.as_ordered_factors():
                if term.is_Pow:
                    term = term.base
                possible_terms.append(term)

    # Replace duplicated expression by new symbol
    new_symbols = {}
    for term in possible_terms:
        if term.is_number:
            continue

        # Convert term to string and replace some characters
        term_str = str(term)
        term_str = term_str.replace("(", "")
        term_str = term_str.replace(")", "")
        term_str = term_str.replace("*", "mul")
        term_str = term_str.replace("/", "div")
        term_str = term_str.replace("+", "add")
        term_str = term_str.replace("-", "sub")
        term_str = term_str.replace(" ", "")

        # if symbol already exist, skip
        if term_str in new_symbols.keys():
            continue

        new_symbols[term_str] = term

    return new_symbols


def main():
    M = pas.mueller()
    theta = Symbol("theta", real=True)
    delta = Symbol("delta", real=True)
    phi1 = Symbol("phi1", real=True)
    phi2 = Symbol("phi2", real=True)
    delta = 2 * pi * 0.25

    # phi1 = 0
    # phi2 = 0

    # Ellipsometry model
    M_obs = pas.polarizer(0) @ pas.retarder(delta, 5 * theta + phi2) @ M @ pas.retarder(delta, theta + phi1) @ pas.polarizer(0)

    # delta = 2 * pi * 0.29
    # M_obs += pas.polarizer(0) @ pas.retarder(delta, 5 * theta + phi2) @ M @ pas.retarder(delta, theta + phi1) @ pas.polarizer(0)
    # delta = 2 * pi * 0.20
    # M_obs += pas.polarizer(0) @ pas.retarder(delta, 5 * theta + phi2) @ M @ pas.retarder(delta, theta + phi1) @ pas.polarizer(0)
    # delta = 2 * pi * 0.22
    # M_obs += pas.polarizer(0) @ pas.retarder(delta, 5 * theta + phi2) @ M @ pas.retarder(delta, theta + phi1) @ pas.polarizer(0)
    # delta = 2 * pi * 0.26
    # M_obs += pas.polarizer(0) @ pas.retarder(delta, 5 * theta + phi2) @ M @ pas.retarder(delta, theta + phi1) @ pas.polarizer(0)

    M_obs_00 = M_obs[0, 0]

    # Get derivative of ln(M_obs_00) by theta
    diff_ln_M_obs_00 = diff(ln(M_obs_00), theta)
    diff_ln_M_obs_00 = factor(expand((diff_ln_M_obs_00)))

    # Get derivative of diff_ln_M_obs_00 by mij
    for mij in M.free_symbols:
        diff_ln_M_obs_00_mij = diff(diff_ln_M_obs_00, mij)
        print(f"d(diff(ln(M_obs_00)), {mij})")
        print(diff_ln_M_obs_00_mij)
        print()

    # exit()

    # Separate by numerator and denominator
    numenator, denominator = fraction(diff_ln_M_obs_00)

    # Extract unique symbols
    numenator_new_symbols = extract_unique_symbols(numenator)
    denominator_new_symbols = extract_unique_symbols(denominator)
    new_symbols = {**numenator_new_symbols, **denominator_new_symbols}
    # Remove Mueller matrix
    for mij in M.free_symbols:
        new_symbols.pop(str(mij), None)

    print("New symbols")
    print(new_symbols)

    # Replace by unique symbols
    for key, value in new_symbols.items():
        new_symbol = Symbol(key, real=True)
        numenator = numenator.subs(value, new_symbol)
        denominator = denominator.subs(value, new_symbol)

    # Decompose coefficients and print
    coffs_numenator = decompose_coefficients(numenator)
    coffs_denominator = decompose_coefficients(denominator)
    print("Numenator")
    for i, coff in enumerate(coffs_numenator):
        print(f"m{i//4}{i%4}: {coff}")

    print("Denominator")
    for i, coff in enumerate(coffs_denominator):
        print(f"m{i//4}{i%4}: {coff}")

    # Generate python functions
    py_funcs_str = ""
    py_funcs_str += "import numpy as np\n"
    py_funcs_str += "\n"

    py_funcs_str += "def diff_ln_M_obs_00(M, theta, phi1, phi2):\n"
    py_funcs_str += "    m00, m01, m02, m03 = M[0]\n"
    py_funcs_str += "    m10, m11, m12, m13 = M[1]\n"
    py_funcs_str += "    m20, m21, m22, m23 = M[2]\n"
    py_funcs_str += "    m30, m31, m32, m33 = M[3]\n"
    py_funcs_str += "\n"
    py_funcs_str += "    sin = np.sin\n"
    py_funcs_str += "    cos = np.cos\n"
    py_funcs_str += "    pi = np.pi\n"
    py_funcs_str += "\n"
    py_funcs_str += "    return "
    py_funcs_str += str(diff_ln_M_obs_00)
    py_funcs_str += "\n"

    py_funcs_str += "def calcNumenatorDenominatorCoffs(theta, phi1, phi2):\n"
    py_funcs_str += "    sin = np.sin\n"
    py_funcs_str += "    cos = np.cos\n"
    py_funcs_str += "\n"
    # expand new symbols
    for key, value in new_symbols.items():
        py_funcs_str += f"    {key} = {value}\n"
    py_funcs_str += "\n"
    py_funcs_str += "    numenator_coffs = np.empty((len(theta), 16), dtype=theta.dtype)\n"
    py_funcs_str += "    denominator_coffs = np.empty((len(theta), 16), dtype=theta.dtype)\n"
    py_funcs_str += "\n"
    for i, coff in enumerate(coffs_numenator):
        py_funcs_str += f"    numenator_coffs[:, {i}] = {coff}\n"
    py_funcs_str += "\n"
    for i, coff in enumerate(coffs_denominator):
        py_funcs_str += f"    denominator_coffs[:, {i}] = {coff}\n"
    py_funcs_str += "\n"
    py_funcs_str += "    return numenator_coffs, denominator_coffs\n"
    py_funcs_str += "\n"

    with open("equations.py", "w") as f:
        py_funcs_str = black.format_str(py_funcs_str, mode=black.Mode(line_length=1000000))
        f.write(py_funcs_str)

    # Generate code with sympy codegen
    replacements, reduced_expressions = cse(coffs_numenator + coffs_denominator, symbols=symbols("tmp0:200"))
    opt = create_expand_pow_optimization(limit=4)

    # Generate cpp functions
    cpp_funcs_str = ""
    cpp_funcs_str += "#include <utility>\n"
    cpp_funcs_str += "#include <Eigen/Core>\n"
    cpp_funcs_str += "#define _USE_MATH_DEFINES\n"
    cpp_funcs_str += "#include <math.h>\n"
    cpp_funcs_str += "\n"

    cpp_funcs_str += "// Calc numerator and denominator of the rational function\n"
    cpp_funcs_str += "std::pair<Eigen::Matrix<float, Eigen::Dynamic, 16>, Eigen::Matrix<float, Eigen::Dynamic, 16>>\n"
    cpp_funcs_str += "calcNumenatorDenominatorCoffs(const Eigen::VectorXf &thetaVector, float phi1, float phi2)\n"
    cpp_funcs_str += "{\n"
    cpp_funcs_str += "    size_t size = thetaVector.size();\n"
    cpp_funcs_str += "    Eigen::Matrix<float, Eigen::Dynamic, 16> numenator_coffs(size, 16);\n"
    cpp_funcs_str += "    Eigen::Matrix<float, Eigen::Dynamic, 16> denominator_coffs(size, 16);\n"
    cpp_funcs_str += "\n"
    # # in for loop
    cpp_funcs_str += "    for (size_t i = 0; i < size; i++)\n"
    cpp_funcs_str += "    {\n"
    cpp_funcs_str += "        float theta = thetaVector(i);\n"
    cpp_funcs_str += "\n"
    # expand new symbols sin, cos
    for key, value in new_symbols.items():
        eq = black.format_str(f"{ccode(opt(value))}", mode=black.Mode(line_length=1000000))
        eq = eq.replace("\n", "")
        cpp_funcs_str += f"        float {key} = {eq};\n"
    cpp_funcs_str += "\n"
    # with reduce symbols
    # tmp
    for key, value in replacements:
        eq = black.format_str(f"{ccode(opt(value))}", mode=black.Mode(line_length=1000000))
        eq = eq.replace("\n", "")
        cpp_funcs_str += f"        float {ccode(key)} = {eq};\n"
    cpp_funcs_str += "\n"
    # numenator
    for i, coff in enumerate(reduced_expressions[: len(coffs_numenator)]):
        eq = black.format_str(f"{ccode(opt(coff))}", mode=black.Mode(line_length=1000000))
        eq = eq.replace("\n", "")
        cpp_funcs_str += f"        numenator_coffs(i, {i}) = {eq};\n"
    cpp_funcs_str += "\n"
    # denominator
    for i, coff in enumerate(reduced_expressions[len(coffs_numenator) :]):
        eq = black.format_str(f"{ccode(opt(coff))}", mode=black.Mode(line_length=1000000))
        eq = eq.replace("\n", "")
        cpp_funcs_str += f"        denominator_coffs(i, {i}) = {eq};\n"
    cpp_funcs_str += "    }\n"
    cpp_funcs_str += "\n"
    cpp_funcs_str += "    return {numenator_coffs, denominator_coffs};\n"
    cpp_funcs_str += "}\n"

    with open("src/equations.cpp", "w") as f:
        f.write(cpp_funcs_str)

    cpp_funcs_str_header = ""
    cpp_funcs_str_header += "#pragma once\n"
    cpp_funcs_str_header += "#include <utility>\n"
    cpp_funcs_str_header += "#include <Eigen/Core>\n"
    cpp_funcs_str_header += "\n"
    cpp_funcs_str_header += "std::pair<Eigen::Matrix<float, Eigen::Dynamic, 16>, Eigen::Matrix<float, Eigen::Dynamic, 16>>\n"
    cpp_funcs_str_header += "calcNumenatorDenominatorCoffs(const Eigen::VectorXf &thetaVector, float phi1, float phi2);\n"

    with open("src/equations.h", "w") as f:
        f.write(cpp_funcs_str_header)


if __name__ == "__main__":
    main()

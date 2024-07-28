from sympy import *
import polanalyser.sympy as pas
import black


def decompose_coefficients(expr: Basic, M: Matrix = pas.mueller()) -> list:
    """Decompose coefficients of expression by Mueller matrix"""
    coffs = []
    for j in range(4):
        for i in range(4):
            if M[j, i] not in expr.free_symbols:
                coff = 0.0
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
    M_obs_00 = M_obs[0, 0]

    # Get derivative of ln(M_obs_00) by theta
    diff_ln_M_obs_00 = diff(ln(M_obs_00), theta)
    diff_ln_M_obs_00 = factor(expand((diff_ln_M_obs_00)))

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
        numenator = numenator.subs(value, Symbol(key, real=True))
        denominator = denominator.subs(value, Symbol(key, real=True))

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

    # Generate cpp functions
    cpp_funcs_str = ""
    cpp_funcs_str += "#include <nanobind/nanobind.h>\n"
    cpp_funcs_str += "#include <nanobind/eigen/dense.h>\n"
    cpp_funcs_str += "\n"

    cpp_funcs_str += "std::pair<Eigen::Matrix<float, Eigen::Dynamic, 16>, Eigen::Matrix<float, Eigen::Dynamic, 16>>\n"
    cpp_funcs_str += "calcNumenatorDenominatorCoffs(const Eigen::VectorXf &theta, float _phi1, float _phi2)\n"
    cpp_funcs_str += "{\n"
    cpp_funcs_str += "    Eigen::Matrix<float, Eigen::Dynamic, 16> numenator_coffs(theta.size(), 16);\n"
    cpp_funcs_str += "    Eigen::Matrix<float, Eigen::Dynamic, 16> denominator_coffs(theta.size(), 16);\n"
    cpp_funcs_str += "\n"
    cpp_funcs_str += "    const auto phi1 = Eigen::VectorXf::Constant(theta.size(), _phi1);\n"
    cpp_funcs_str += "    const auto phi2 = Eigen::VectorXf::Constant(theta.size(), _phi2);\n"
    cpp_funcs_str += "\n"
    for key, value in new_symbols.items():
        # Replace sin and cos by array version for Eigen
        # sin(2 * phi1 + 2 * theta) -> (2 * phi1 + 2 * theta).array().sin()
        has_sin = str(value)[:3] == "sin"
        has_cos = str(value)[:3] == "cos"
        value = str(value).replace("sin", "").replace("cos", "")
        if has_sin and not has_cos:
            eq = f"{key} = {value}.array().sin()"
        elif has_cos and not has_sin:
            eq = f"{key} = {value}.array().cos()"
        else:
            raise
        eq = black.format_str(eq, mode=black.Mode(line_length=1000000))
        eq = eq.replace("\n", "")
        cpp_funcs_str += f"    const auto {eq};\n"
    cpp_funcs_str += "\n"
    for i, coff in enumerate(coffs_numenator):
        try:
            num = float(coff)
            cpp_funcs_str += f"    numenator_coffs.col({i}) = Eigen::VectorXf::Constant(theta.size(), {num});\n"
        except:
            # Replace **2 by .square()
            coff = str(coff).replace("**2", ".square()")
            eq = f"numenator_coffs.col({i}) = {coff}"
            eq = black.format_str(eq, mode=black.Mode(line_length=1000000))
            eq = eq.replace("\n", "")
            cpp_funcs_str += f"    {eq};\n"

    cpp_funcs_str += "\n"
    for i, coff in enumerate(coffs_denominator):
        try:
            num = float(coff)
            cpp_funcs_str += f"    denominator_coffs.col({i}) = Eigen::VectorXf::Constant(theta.size(), {num});\n"
        except:
            # Replace **2 by .square()
            coff = str(coff).replace("**2", ".square()")
            eq = f"denominator_coffs.col({i}) = {coff}"
            eq = black.format_str(eq, mode=black.Mode(line_length=1000000))
            eq = eq.replace("\n", "")
            cpp_funcs_str += f"    {eq};\n"
    cpp_funcs_str += "\n"
    cpp_funcs_str += "    return {numenator_coffs, denominator_coffs};\n"
    cpp_funcs_str += "}\n"

    with open("equations.h", "w") as f:
        f.write(cpp_funcs_str)


if __name__ == "__main__":
    main()

import numpy as np

_Factlist = [1.0, 1.0]


def _calc_factlist(nn):
    r"""
    Function calculates a list of precomputed factorials in order to
    massively accelerate future calculations of the various
    coefficients.
    Parameters
    ==========
    nn : integer
        Highest factorial to be computed.
    Returns
    =======
    list of reals :
        The list of precomputed factorials.
    Examples
    ========
    Calculate list of factorials::
        sage: from sage.functions.wigner import _calc_factlist
        sage: _calc_factlist(10)
        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    """
    if nn >= len(_Factlist):
        for ii in range(len(_Factlist), int(nn + 1)):
            _Factlist.append(_Factlist[ii - 1] * ii)
    return _Factlist[:int(nn) + 1]


def wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3):
    r"""
    Calculate the Wigner 3j symbol `\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`.
    Parameters
    ==========
    j_1, j_2, j_3, m_1, m_2, m_3 :
        Integer or half integer.
    Returns
    =======
    Rational number times the square root of a rational number.
    Examples
    ========
    >>> from sympy.physics.wigner import wigner_3j
    >>> wigner_3j(2, 6, 4, 0, 0, 0)
    sqrt(715)/143
    >>> wigner_3j(2, 6, 4, 0, 0, 1)
    0
    It is an error to have arguments that are not integer or half
    integer values::
        sage: wigner_3j(2.1, 6, 4, 0, 0, 0)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer
        sage: wigner_3j(2, 6, 4, 1, 0, -1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer or half integer
    Notes
    =====
    The Wigner 3j symbol obeys the following symmetry rules:
    - invariant under any permutation of the columns (with the
      exception of a sign change where `J:=j_1+j_2+j_3`):
      .. math::
         \begin{aligned}
         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
          &=\operatorname{Wigner3j}(j_3,j_1,j_2,m_3,m_1,m_2) \\
          &=\operatorname{Wigner3j}(j_2,j_3,j_1,m_2,m_3,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_3,j_2,j_1,m_3,m_2,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_1,j_3,j_2,m_1,m_3,m_2) \\
          &=(-1)^J \operatorname{Wigner3j}(j_2,j_1,j_3,m_2,m_1,m_3)
         \end{aligned}
    - invariant under space inflection, i.e.
      .. math::
         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
         =(-1)^J \operatorname{Wigner3j}(j_1,j_2,j_3,-m_1,-m_2,-m_3)
    - symmetric with respect to the 72 additional symmetries based on
      the work by [Regge58]_
    - zero for `j_1`, `j_2`, `j_3` not fulfilling triangle relation
    - zero for `m_1 + m_2 + m_3 \neq 0`
    - zero for violating any one of the conditions
      `j_1 \ge |m_1|`,  `j_2 \ge |m_2|`,  `j_3 \ge |m_3|`
    Algorithm
    =========
    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.
    Authors
    =======
    - Jens Rasch (2009-03-24): initial version
    """
    if int(j_1 * 2) != j_1 * 2 or int(j_2 * 2) != j_2 * 2 or \
            int(j_3 * 2) != j_3 * 2:
        raise ValueError("j values must be integer or half integer")
    if int(m_1 * 2) != m_1 * 2 or int(m_2 * 2) != m_2 * 2 or \
            int(m_3 * 2) != m_3 * 2:
        raise ValueError("m values must be integer or half integer")
    if m_1 + m_2 + m_3 != 0:
        return 0
    prefid = 1 if int(j_1 - j_2 - m_3) % 2 == 0 else -1
    m_3 = -m_3
    a1 = j_1 + j_2 - j_3
    if a1 < 0:
        return 0
    a2 = j_1 - j_2 + j_3
    if a2 < 0:
        return 0
    a3 = -j_1 + j_2 + j_3
    if a3 < 0:
        return 0
    if (abs(m_1) > j_1) or (abs(m_2) > j_2) or (abs(m_3) > j_3):
        return 0

    maxfact = max(j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2),
                  j_3 + abs(m_3))
    _calc_factlist(int(maxfact))

    argsqrt =  _Factlist[int(j_1 + j_2 - j_3)] * \
                     _Factlist[int(j_1 - j_2 + j_3)] * \
                     _Factlist[int(-j_1 + j_2 + j_3)] * \
                     _Factlist[int(j_1 - m_1)] * \
                     _Factlist[int(j_1 + m_1)] * \
                     _Factlist[int(j_2 - m_2)] * \
                     _Factlist[int(j_2 + m_2)] * \
                     _Factlist[int(j_3 - m_3)] * \
                     _Factlist[int(j_3 + m_3)] / \
        _Factlist[int(j_1 + j_2 + j_3 + 1)]

    ressqrt = np.sqrt(argsqrt)

    imin = max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
    imax = min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * \
            _Factlist[int(ii + j_3 - j_1 - m_2)] * \
            _Factlist[int(j_2 + m_2 - ii)] * \
            _Factlist[int(j_1 - ii - m_1)] * \
            _Factlist[int(ii + j_3 - j_2 + m_1)] * \
            _Factlist[int(j_1 + j_2 - j_3 - ii)]
        sumres = sumres + (2 * (ii % 2) - 1) / den

    res = ressqrt * sumres * prefid
    #print('sdf', ressqrt, sumres, prefid, res)
    return res


def CG(j_1, j_2, j_3, m_1, m_2, m_3):
    r"""
    Calculates the Clebsch-Gordan coefficient.
    `\left\langle j_1 m_1 \; j_2 m_2 | j_3 m_3 \right\rangle`.
    The reference for this function is [Edmonds74]_.
    Parameters
    ==========
    j_1, j_2, j_3, m_1, m_2, m_3 :
        Integer or half integer.
    Returns
    =======
    Rational number times the square root of a rational number.
    Examples
    ========
    >>> from sympy import S
    >>> from sympy.physics.wigner import clebsch_gordan
    >>> clebsch_gordan(S(3)/2, S(1)/2, 2, S(3)/2, S(1)/2, 2)
    1
    >>> clebsch_gordan(S(3)/2, S(1)/2, 1, S(3)/2, -S(1)/2, 1)
    sqrt(3)/2
    >>> clebsch_gordan(S(3)/2, S(1)/2, 1, -S(1)/2, S(1)/2, 0)
    -sqrt(2)/2
    Notes
    =====
    The Clebsch-Gordan coefficient will be evaluated via its relation
    to Wigner 3j symbols:
    .. math::
        \left\langle j_1 m_1 \; j_2 m_2 | j_3 m_3 \right\rangle
        =(-1)^{j_1-j_2+m_3} \sqrt{2j_3+1}
        \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,-m_3)
    See also the documentation on Wigner 3j symbols which exhibit much
    higher symmetry relations than the Clebsch-Gordan coefficient.
    Authors
    =======
    - Jens Rasch (2009-03-24): initial version
    """
    res = (2*(int(j_1 - j_2 + m_3)%2)-1) * np.sqrt(2 * j_3 + 1) * \
        wigner_3j(j_1, j_2, j_3, m_1, m_2, -m_3)
    return res

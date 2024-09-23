# define a random distribution from which width parameters are chose if and only if
# the binary intertwining operation yields values outside the acceptable interval
loc, scale = 1.3, 100.5  # TODO, loc should be chosen s.t. the Gaussian having the same width as the exponential prop. density

# within this interval, the clipped distribution is squeezed, and here we set this with the
# width bounds as sepcified in NextToNewest...
width_bnds = [np.min(initialGridBounds), np.max(initialGridBounds)]

a_transformed, b_transformed = (width_bnds[0] - loc) / scale, (width_bnds[1] -
                                                               loc) / scale
rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
x = np.linspace(truncnorm.ppf(0.01, width_bnds[0], width_bnds[1]),
                truncnorm.ppf(1, width_bnds[0], width_bnds[1]), 100)

r = rv.rvs(size=10000)

if dbg:
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.set_xlim(width_bnds[0], width_bnds[1])
    ax.legend(loc='best', frameon=False)

    fig.savefig("default_breeding_widths.pdf")


# each instantiation of the crossover function probes into the above-sampled set of
# default values: def1=rv.rvs()
def intertwining(p1,
                 p2,
                 def1,
                 def2,
                 mutation_rate=0.0,
                 wMin=0.00001,
                 wMax=920.,
                 dbg=False,
                 method='1point'):

    Bp1 = float_to_bin(p1)
    Bp2 = float_to_bin(p2)

    defaul = False

    assert len(Bp1) == len(Bp2)
    assert mutation_rate < 1

    mutationMask = np.random.choice(2,
                                    p=[1 - mutation_rate, mutation_rate],
                                    size=len(Bp1))

    if method == '1point':

        pivot = np.random.randint(0, len(Bp1))

        Bchild1 = Bp1[:pivot] + Bp2[pivot:]
        Bchild2 = Bp2[:pivot] + Bp1[pivot:]

        Bchild2mutated = ''.join(
            (mutationMask | np.array(list(Bchild2)).astype(int)).astype(str))
        Bchild1mutated = ''.join(
            (mutationMask | np.array(list(Bchild1)).astype(int)).astype(str))

        Fc1 = np.abs(bin_to_float(Bchild1mutated))
        Fc2 = np.abs(bin_to_float(Bchild2mutated))

        # Check for out-of-range or NaN values
        # the defaults are drawn from a clipped normal distribution which is
        # defined for the system the basis is optimized
        Fc1 = def1 if (np.isnan(Fc1) or Fc1 < wMin or Fc1 > wMax) else Fc1
        Fc2 = def2 if (np.isnan(Fc2) or Fc2 < wMin or Fc2 > wMax) else Fc2

    elif method == '2point':

        # Determine two pivot points for the multi-point crossover
        pivot1 = np.random.randint(0, int(len(Bp1) / 2))
        pivot2 = np.random.randint(pivot1 + 1, len(Bp1))

        # Swap pivot points if pivot2 is less than pivot1
        if pivot2 < pivot1:
            pivot1, pivot2 = pivot2, pivot1

        # Perform crossover using the multi-point method
        Bchild1 = Bp1[:pivot1] + Bp2[pivot1:pivot2] + Bp1[pivot2:]
        Bchild2 = Bp2[:pivot1] + Bp1[pivot1:pivot2] + Bp2[pivot2:]

        # Apply mutation
        Bchild1mutated = ''.join(
            (mutationMask | np.array(list(Bchild1)).astype(int)).astype(str))
        Bchild2mutated = ''.join(
            (mutationMask | np.array(list(Bchild2)).astype(int)).astype(str))

        # Convert binary strings to floating-point values
        Fc1 = np.abs(bin_to_float(Bchild1mutated))
        Fc2 = np.abs(bin_to_float(Bchild2mutated))

        # Check for out-of-range or NaN values
        # the defaults are drawn from a clipped normal distribution which is
        # defined for the system the basis is optimized
        Fc1 = def1 if (np.isnan(Fc1) or Fc1 < wMin or Fc1 > wMax) else Fc1
        Fc2 = def2 if (np.isnan(Fc2) or Fc2 < wMin or Fc2 > wMax) else Fc2

    elif method == '4point':

        # Determine four pivot points for the four-point crossover
        pivot1 = np.random.randint(0, len(Bp1))
        pivot2 = np.random.randint(pivot1 + 1, len(Bp1))
        pivot3 = np.random.randint(pivot2 + 1, len(Bp1))
        pivot4 = np.random.randint(pivot3 + 1, len(Bp1))

        # Perform crossover using the four-point method
        Bchild1 = Bp1[:pivot1] + Bp2[pivot1:pivot2] + Bp1[pivot2:pivot3] + Bp2[
            pivot3:pivot4] + Bp1[pivot4:]
        Bchild2 = Bp2[:pivot1] + Bp1[pivot1:pivot2] + Bp2[pivot2:pivot3] + Bp1[
            pivot3:pivot4] + Bp2[pivot4:]

        # Apply mutation
        Bchild1mutated = ''.join(
            (mutationMask | np.array(list(Bchild1)).astype(int)).astype(str))
        Bchild2mutated = ''.join(
            (mutationMask | np.array(list(Bchild2)).astype(int)).astype(str))

        # Convert binary strings to floating-point values
        Fc1 = np.abs(bin_to_float(Bchild1mutated))
        Fc2 = np.abs(bin_to_float(Bchild2mutated))

        # Check for out-of-range or NaN values
        # the defaults are drawn from a clipped normal distribution which is
        # defined for the system the basis is optimized
        Fc1 = def1 if (np.isnan(Fc1) or Fc1 < wMin or Fc1 > wMax) else Fc1
        Fc2 = def2 if (np.isnan(Fc2) or Fc2 < wMin or Fc2 > wMax) else Fc2

    elif method == 'uniform':

        # Perform uniform crossover
        Bchild1 = ''
        Bchild2 = ''
        for i in range(len(Bp1)):
            if np.random.rand() < 0.5:
                Bchild1 += Bp1[i]
                Bchild2 += Bp2[i]
            else:
                Bchild1 += Bp2[i]
                Bchild2 += Bp1[i]

        # Apply mutation
        Bchild1mutated = ''.join(
            (mutationMask | np.array(list(Bchild1)).astype(int)).astype(str))
        Bchild2mutated = ''.join(
            (mutationMask | np.array(list(Bchild2)).astype(int)).astype(str))

        # Convert binary strings to floating-point values
        Fc1 = np.abs(bin_to_float(Bchild1mutated))
        Fc2 = np.abs(bin_to_float(Bchild2mutated))

        # Check for out-of-range or NaN values
        # the defaults are drawn from a clipped normal distribution which is
        # defined for the system the basis is optimized
        Fc1 = def1 if (np.isnan(Fc1) or Fc1 < wMin or Fc1 > wMax) else Fc1
        Fc2 = def2 if (np.isnan(Fc2) or Fc2 < wMin or Fc2 > wMax) else Fc2

    else:
        print('unspecified intertwining method.')
        exit()

    if (dbg | np.isnan(Fc1) | np.isnan(Fc2)):
        print('parents (binary)        :%12.4f%12.4f' % (p1, p2))
        print('parents (decimal)       :', Bp1, ';;', Bp2)
        print('children (binary)       :', Bchild1, ';;', Bchild2)
        print('children (decimal)      :%12.4f%12.4f' % (Fc1, Fc2))

    return Fc1, Fc2, defaul
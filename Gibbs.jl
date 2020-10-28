
# objective function: uses penalty method to ensure constraints are satisfied -
function _objective_function_gibbs_composition(x,P_ref,T_K,V_dot,AM,bV,dG_formation)

    R = 8.314               # L kPa K^-1 mol^-1
    x = abs.(x)

    # convert to dimensionless -
    dG_formation = (1/(R*T_K)).*dG_formation;

    # initialize some stuff -
    weight_factor = 100000.0
    number_of_species = length(x)
    mol_fraction_array = Array{Float64,1}(undef, number_of_species)
    gibbs_energy_array = Array{Float64,1}(undef, number_of_species)

    # compute the mol fraction array -
    mol_total = sum(x)
    for species_index = 1:number_of_species
        mol_value = x[species_index]
        mol_fraction = mol_value/mol_total
        mol_fraction_array[species_index] = mol_fraction
    end

    # compute PBAR -
    PBAR = ((R*T_K)/(V_dot*P_ref))*sum(x)

    # compute the Gibbs energy array -
    for species_index = 1:number_of_species
        gibbs_term = x[species_index]*(dG_formation[species_index]+log(mol_fraction_array[species_index]*PBAR))
        gibbs_energy_array[species_index] = gibbs_term
    end

    # compute the error terms (penalty)
    error_terms = AM*x - bV

    # compute overall objective value -
    objective_value = sum(gibbs_energy_array) + weight_factor*(transpose(error_terms)*error_terms)

    # return -
    return objective_value
end

# main method: minimizes the Gibbs energy by finding the equlibrium compostion -
function min_direct_gibbs_composition()

    # setup problem/calculation -
    R = 8.314               # L kPa K^-1 mol^-1
    T_K = (700 + 273.15)    # K
    P_ref = 100000.0        # kPa (1000 bar?)
    V_dot = 547.22          # L/s (given)

    # setup the atom array -
    A = [0 2 0; 2 0 0; 1 3 0; 0 2 1; 0 0 2]
    AM = transpose(A)

    # compute the bV -
    ndot_initial_in = [
        1e-10       ;   # 1  H2
        1e-10       ;   # 2  N2
        30.5211     ;   # 3  NH3
        0.0578212   ;   # 4  H20
        1e-10       ;   # 5  O2
    ];
    bV = AM*ndot_initial_in

    # what are the dGs of formation
    dG_formation = [
        1e-10       ;   # 1  H2
        1e-10       ;   # 2  N2
        -1.644      ;   # 3  NH3
        -228.572    ;   # 4  H20
        1e-10       ;   # 5  O2
    ];

    # setup initial condition -
    initial_condition = ndot_initial_in

    # setup objective function -
    OF(n) = _objective_function_gibbs_composition(n,P_ref, T_K, V_dot, AM, bV, dG_formation)

    # setup contraints -
    lower = zeros(5)
    upper = 1000*ones(5)

    # call the optimizer -
    opt_result = optimize(OF,lower,upper,initial_condition,Fminbox(BFGS()))

    # what is Vstar?
    ndot_out = Optim.minimizer(opt_result)

    # return -
    return ndot_out
end

# n_dot = main_direct_gibbs_composition()
# 𝝐 = main_direct_gibbs_extent()

# test -
# T_K = (700 + 273.15)
# R = 8.314
# Vdot = 547.2
# P_e = ((R*T_K)/(Vdot))*sum(n_dot)
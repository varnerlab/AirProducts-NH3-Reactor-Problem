function _objective_function_gibbs_extent(x, initial_composition, stochiometric_matrix, P_ref,T_K,V_dot,dG_formation)

    R = 8.314               # L kPa K^-1 mol^-1
    𝝐 = abs(x[1])                # get the extent of reaction

    # initialize some stuff -
    number_of_species = length(initial_composition)
    mol_fraction_array = Array{Float64,1}(undef, number_of_species)
    gibbs_energy_array = Array{Float64,1}(undef, number_of_species) 

    # compute the mol fraction array -
    new_composition = initial_composition + stochiometric_matrix*𝝐
    mol_total = sum(new_composition)
    for species_index = 1:number_of_species
        mol_value = new_composition[species_index]
        mol_fraction = mol_value/mol_total
        mol_fraction_array[species_index] = mol_fraction
    end

    # compute PBAR -
    PBAR = ((R*T_K)/(V_dot*P_ref))*sum(new_composition)

    # compute the dG of reaction term -
    tmp_array = Array{Float64,1}()
    for species_index = 1:number_of_species
        tmp_term = stochiometric_matrix[species_index]*dG_formation[species_index]
        push!(tmp_array,tmp_term)
    end
    dG_reaction = (1/(R*T_K))*(sum(tmp_array))

    # compute the fugacity terms -
    activity_term_array = Array{Float64,1}()
    for species_index = 1:number_of_species
        term = new_composition[species_index]*log(mol_fraction_array[species_index]*PBAR)
        push!(activity_term_array, term)
    end
    
    # compute obj value -
    obj_value = 𝝐*dG_reaction+sum(activity_term_array)

    # return -
    return obj_value
end

# main method: minimizes the Gibbs energy by finding the equlibrium extent of reaction
function min_direct_gibbs_extent()

    # setup problem/calculation -
    R = 8.314               # L kPa K^-1 mol^-1
    T_K = (700 + 273.15)    # K
    P_ref = 100000.0        # kPa (1000 bar?)
    V_dot = 547.22          # L/s (given)

    # what is the input mol flow rates -
    ndot_initial_in = [
        1e-10       ;   # 1  H2
        1e-10       ;   # 2  N2
        30.5211     ;   # 3  NH3
        0.0578212   ;   # 4  H20
        1e-10       ;   # 5  O2
    ];

    # r1: 2*NH3 = N2+3*H2
    stochiometric_matrix = [
        
        3.0         ;   # 1  H2
        1.0         ;   # 2  N2
        -2.0        ;   # 3  NH3
        0.0         ;   # 4  H20
        0.0         ;   # 5  O2
    ]

    # what are the dGs of formation?
    dG_formation = [
        1e-10       ;   # 1  H2
        1e-10       ;   # 2  N2
        -1.644      ;   # 3  NH3
        -228.572    ;   # 4  H20
        1e-10       ;   # 5  O2
    ];

    # setup initial condition -
    initial_composition = ndot_initial_in

    # setup objective function -
    OF(𝝐) = _objective_function_gibbs_extent(𝝐, initial_composition, stochiometric_matrix, P_ref,T_K,V_dot, dG_formation)

    # call the optimizer -
    lower = [0]
    upper = [30.5211/2.0]
    opt_result = optimize(OF,lower,upper,[0.1],Fminbox(BFGS()))

    # what is Vstar?
    𝝐 = Optim.minimizer(opt_result)[1]

    # return -
    return 𝝐
end
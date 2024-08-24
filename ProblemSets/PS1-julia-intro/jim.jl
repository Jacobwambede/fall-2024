using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions
# Jacob  Wambede
# 9/15/2021
# PS1: Julia Introduction
# Econometric III: ECON 6343
# Question 1
function q1()
    Random.seed!(1234)
    
    # a. Create matrices
    A = rand(Uniform(-5, 10), 10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    C = [A[1:5, 1:5] B[1:5, 6:7]]
    D = [a <= 0 ? a : 0 for a in A]

    # b. Number of elements in A
    println("Number of elements in A: ", length(A))

    # c. Number of unique elements in D
    println("Number of unique elements in D: ", length(unique(D)))

    # d. Create matrix E (vec of B)
    E = vec(B)

    # e. Create 3D array F
    F = cat(A, B, dims=3)

    # f. Permute dimensions of F
    F = permutedims(F, (3, 1, 2))

    # g. Kronecker product
    G = kron(B, C)

    # h. Save matrices to JLD file
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

    # i. Save subset of matrices
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)

    # j. Export C as CSV
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))

    # k. Export D as tab-delimited dat file
    CSV.write("Dmatrix.dat", DataFrame(D, :auto), delim='\t')

    return A, B, C, D
end

# Question 2
function q2(A, B, C)
    # a. Element-wise product of A and B
    AB = A .* B
    AB2 = A .* B  # Same result without loop

    # b. Elements of C between -5 and 5
    Cprime = [c for c in C if -5 <= c <= 5]
    Cprime2 = filter(c -> -5 <= c <= 5, vec(C))

    # c. Create 3D array X
    N, K, T = 15169, 6, 5
    X = Array{Float64}(undef, N, K, T)
    
    for t in 1:T
        X[:, 1, t] .= 1  # Intercept
        X[:, 2, t] = rand(Bernoulli(0.75 * (6 - t) / 5), N)
        X[:, 3, t] = rand(Normal(15 + t - 1, 5 * (t - 1)), N)
        X[:, 4, t] = rand(Normal(π * (6 - t) / 3, 1/ℯ), N)
        X[:, 5, t] = rand(Binomial(20, 0.6), N)
        X[:, 6, t] = rand(Binomial(20, 0.5), N)
    end

    # d. Create β matrix
    β = [
        [1 + 0.25*(t-1) for t in 1:T]';
        [log(t) for t in 1:T]';
        [-sqrt(t) for t in 1:T]';
        [exp(t) - exp(t+1) for t in 1:T]';
        [t for t in 1:T]';
        [t/3 for t in 1:T]'
    ]

    # e. Create Y matrix
    ε = rand(Normal(0, 0.36), N, T)
    Y = [sum(X[i, :, t] .* β[:, t]) + ε[i, t] for i in 1:N, t in 1:T]
end

# Question 3
function q3()
    # a. Import and process data
    nlsw88 = CSV.read("nlsw88.csv", DataFrame)
    nlsw88 = coalesce.(nlsw88, missing)  # Convert "" to missing
    CSV.write("nlsw88_processed.csv", nlsw88)

    # b. Percentage never married and college graduates
    println("Never married: ", mean(nlsw88.never_married .== 1) * 100, "%")
    println("College graduates: ", mean(nlsw88.collgrad .== 1) * 100, "%")

    # c. Race distribution
    println(prop(freqtable(nlsw88.race)))

    # d. Summary statistics
    summarystats = describe(nlsw88)
    println("Missing grade observations: ", sum(ismissing.(nlsw88.grade)))

    # e. Joint distribution of industry and occupation
    println(freqtable(nlsw88.industry, nlsw88.occupation))

    # f. Mean wage over industry and occupation
    wage_by_ind_occ = combine(groupby(nlsw88, [:industry, :occupation]), :wage => mean)
    println(wage_by_ind_occ)
end

# Question 4
function q4()
    # a. Load matrices
    data = load("firstmatrix.jld")
    A, B = data["A"], data["B"]

    # b-e. Define matrixops function
    function matrixops(A, B)
        # This function performs matrix operations on two input matrices
        if size(A) != size(B)
            error("inputs must have the same size")
        end
        return A .* B, A' * B, sum(A + B)
    end

    # d. Evaluate matrixops with A and B
    result = matrixops(A, B)
    println("Result of matrixops(A, B): ", result)

    # f. Evaluate matrixops with C and D (This will raise an error)
    # Uncomment to test: matrixops(data["C"], data["D"])

    # g. Evaluate matrixops with ttl_exp and wage
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)
    ttl_exp = convert(Array, nlsw88.ttl_exp)
    wage = convert(Array, nlsw88.wage)
    result_nlsw = matrixops(ttl_exp, wage)
    println("Result of matrixops(ttl_exp, wage): ", result_nlsw)
end

# Main execution
A, B, C, D = q1()
q2(A, B, C)
q3()
q4()

# Unit tests (basic examples, expand as needed)
@assert size(A) == (10, 7) "A should be 10x7"
@assert size(B) == (10, 7) "B should be 10x7"
@assert size(C) == (5, 7) "C should be 5x7"
@assert all(D .<= 0) "All elements in D should be <= 0"
println("All tests passed!")
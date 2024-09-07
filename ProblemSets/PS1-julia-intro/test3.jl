using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions, Test

# Define matrixops function in the global scope
function matrixops(A, B)
    # This function performs matrix operations on two input matrices
    # Inputs: A, B - matrices of the same size
    # Outputs: (1) Element-wise product of A and B
    #          (2) Matrix product of A' and B
    #          (3) Sum of all elements in A + B

    # Check if inputs have the same size
    if size(A) != size(B)
        error("inputs must have the same size")
    end

    return A .* B, A' * B, sum(A + B)
end

# Question 1
function q1()
    Random.seed!(1234)

    # (a) Create matrices
    A = rand(Uniform(-5, 10), 10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    C = [A[1:5, 1:5] B[1:5, 6:7]]
    D = [A[i,j] <= 0 ? A[i,j] : 0 for i in 1:10, j in 1:7]

    # (b) Number of elements in A
    println("Number of elements in A: ", length(A))

    # (c) Number of unique elements in D
    println("Number of unique elements in D: ", length(unique(D)))

    # (d) Create matrix E
    E = reshape(B, :, 1)

    # (e) Create 3D array F
    F = cat(A, B, dims=3)

    # (f) Twist F
    F = permutedims(F, (3, 1, 2))

    # (g) Kronecker product
    G = kron(B, C)
    
    # (h) Save matrices to .jld file
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

    # (i) Save matrices A, B, C, D to .jld file
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)

    # (j) Export C as .csv file
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))

    # (k) Export D as tab-delimited .dat file
    CSV.write("Dmatrix.dat", DataFrame(D, :auto), delim='\t')

    return A, B, C, D
end

# Question 2
function q2(A, B, C)
    # (a) Element-wise product of A and B
    AB = A .* B
    AB2 = A .* B  # This accomplishes the same without a loop

    # (b) Create Cprime
    Cprime = [x for x in C if -5 <= x <= 5]
    Cprime2 = filter(x -> -5 <= x <= 5, C[:])

    # (c) Create 3D array X
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

    # (d) Create β matrix
    β = [t == 1 ? 1 : 1 + 0.25*(t-1) for t in 1:T, _ in 1:K]'
    β[2, :] = log.(1:T)
    β[3, :] = -sqrt.(1:T)
    β[4, :] = exp.(1:T) - exp.(2:T+1)
    β[5, :] = 1:T
    β[6, :] = (1:T) ./ 3

    # (e) Create Y matrix
    ε = rand(Normal(0, 0.36), N, T)
    Y = [sum(X[n, :, t] .* β[:, t]) + ε[n, t] for n in 1:N, t in 1:T]

    return AB, AB2, Cprime, Cprime2, X, β, Y
end

# Question 3
function q3()
    # (a) Import and process data
    nlsw88 = CSV.read("nlsw88.csv", DataFrame)
    nlsw88 = coalesce.(nlsw88, missing)  # Convert "" to missing
    CSV.write("nlsw88_processed.csv", nlsw88)

    # (b) Percentage never married and college graduates
    pct_never_married = mean(nlsw88.never_married .== 1) * 100
    pct_college_grad = mean(nlsw88.collgrad .== 1) * 100
    println("Percentage never married: ", pct_never_married)
    println("Percentage college graduates: ", pct_college_grad)

    # (c) Percentage in each race category
    race_dist = freqtable(nlsw88.race)
    println("Race distribution: ", prop(race_dist))

    # (d) Summary statistics
    summarystats = describe(nlsw88)
    println("Number of missing grade observations: ", sum(ismissing.(nlsw88.grade)))

    # (e) Joint distribution of industry and occupation
    industry_occupation = freqtable(nlsw88.industry, nlsw88.occupation)
    println("Joint distribution of industry and occupation: ", prop(industry_occupation))

    # (f) Mean wage over industry and occupation
    wage_by_ind_occ = combine(groupby(nlsw88, [:industry, :occupation]), :wage => mean)
    println("Mean wage by industry and occupation: ", wage_by_ind_occ)

    return nlsw88, summarystats, industry_occupation, wage_by_ind_occ
end

# Question 4
function q4()
    # (a) Load firstmatrix.jld
    data = load("firstmatrix.jld")
    A, B = data["A"], data["B"]

    # (d) Evaluate matrixops with A and B
    result1, result2, result3 = matrixops(A, B)
    println("Element-wise product: ", result1)
    println("A' * B: ", result2)
    println("Sum of A + B: ", result3)

    # (g) Evaluate matrixops with ttl_exp and wage from nlsw88
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)
    ttl_exp = Matrix(nlsw88.ttl_exp)
    wage = Matrix(nlsw88.wage)
    result1, result2, result3 = matrixops(ttl_exp, wage)
    println("Element-wise product of ttl_exp and wage: ", result1)
    println("ttl_exp' * wage: ", result2)
    println("Sum of ttl_exp + wage: ", result3)

    return A, B, ttl_exp, wage
end

# Execute all functions
A, B, C, D = q1()
AB, AB2, Cprime, Cprime2, X, β, Y = q2(A, B, C)
nlsw88, summarystats, industry_occupation, wage_by_ind_occ = q3()
A_q4, B_q4, ttl_exp, wage = q4()

# Question 5: Unit Tests
@testset "Problem Set 1 Tests" begin
    @testset "q1 Tests" begin
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)
        @test size(D) == (10, 7)
        @test all(D .<= 0)
    end

    @testset "q2 Tests" begin
        @test size(AB) == size(A)
        @test size(X) == (15169, 6, 5)
        @test size(β) == (6, 5)
        @test size(Y) == (15169, 5)
    end

    @testset "q3 Tests" begin
        @test nrow(nlsw88) > 0
        @test ncol(nlsw88) > 0
        @test size(summarystats, 1) == ncol(nlsw88)
        @test size(industry_occupation) == (12, 12)  # Assuming 12 industries and 12 occupations
    end

    @testset "q4 Tests" begin
        @test size(A_q4) == (10, 7)
        @test size(B_q4) == (10, 7)
        result1, result2, result3 = matrixops(A_q4, B_q4)
        @test size(result1) == (10, 7)
        @test size(result2) == (7, 7)
        @test isa(result3, Number)
    end

    @testset "matrixops Tests" begin
        A_test = rand(3, 3)
        B_test = rand(3, 3)
        result1, result2, result3 = matrixops(A_test, B_test)
        @test size(result1) == (3, 3)
        @test size(result2) == (3, 3)
        @test isa(result3, Number)
        @test_throws ErrorException matrixops(rand(2, 2), rand(3, 3))
    end
end

println("All tasks completed successfully!")
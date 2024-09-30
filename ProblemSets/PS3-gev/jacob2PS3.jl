using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables
using Test

function allwrap()
    # Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    # 1. Multinomial Logit Model

    function mnl_loglikelihood(β, X, Z, y)
        n, p = size(X)
        J = size(Z, 2)
        
        ll = 0.0
        for i in 1:n
            denom = 1.0
            for j in 1:(J-1)
                denom += exp(X[i,:]' * β[((j-1)*p+1):(j*p)] + β[end] * (Z[i,j] - Z[i,J]))
            end
            
            if y[i] == J
                ll += -log(denom)
            else
                ll += X[i,:]' * β[((y[i]-1)*p+1):(y[i]*p)] + β[end] * (Z[i,y[i]] - Z[i,J]) - log(denom)
            end
        end
        
        return -ll  # Negative log-likelihood for minimization
    end

    function estimate_mnl(X, Z, y)
        n, p = size(X)
        J = size(Z, 2)
        
        # Initialize parameters
        β_init = vcat(zeros((J-1)*p), [0.0])
        
        # Optimize
        result = optimize(β -> mnl_loglikelihood(β, X, Z, y), β_init, BFGS())
        
        return Optim.minimizer(result)
    end

    # Estimate multinomial logit model
    β_mnl = estimate_mnl(X, Z, y)

    println("Multinomial Logit Estimates:")
    println(β_mnl)

    # 2. Interpretation of γ̂
    γ̂ = β_mnl[end]
    println("\nInterpretation of γ̂:")
    println("γ̂ = ", γ̂)
    println("This coefficient represents the effect of the difference in log wages on the probability of choosing an occupation.")
    println("A positive γ̂ indicates that individuals are more likely to choose occupations with higher wages.")

    # 3. Nested Logit Model

    function nested_logit_loglikelihood(θ, X, Z, y)
        n, k = size(X)
        J = size(Z, 2)
        
        β_WC = θ[1:k]
        β_BC = θ[(k+1):(2k)]
        λ_WC = θ[2k+1]
        λ_BC = θ[2k+2]
        γ = θ[end]
        
        ll = 0.0
        for i in 1:n
            if y[i] in [1, 2, 3]  # White Collar
                num = exp((dot(X[i,:], β_WC) + γ*(Z[i,y[i]] - Z[i,J])) / λ_WC)
                denom_WC = sum(exp((dot(X[i,:], β_WC) + γ*(Z[i,j] - Z[i,J])) / λ_WC) for j in 1:3)
                prob_nest = denom_WC^λ_WC / (1 + denom_WC^λ_WC + (sum(exp((dot(X[i,:], β_BC) + γ*(Z[i,j] - Z[i,J])) / λ_BC) for j in 4:7))^λ_BC)
                ll += log(num / denom_WC) + log(prob_nest)
            elseif y[i] in [4, 5, 6, 7]  # Blue Collar
                num = exp((dot(X[i,:], β_BC) + γ*(Z[i,y[i]] - Z[i,J])) / λ_BC)
                denom_BC = sum(exp((dot(X[i,:], β_BC) + γ*(Z[i,j] - Z[i,J])) / λ_BC) for j in 4:7)
                prob_nest = denom_BC^λ_BC / (1 + (sum(exp((dot(X[i,:], β_WC) + γ*(Z[i,j] - Z[i,J])) / λ_WC) for j in 1:3))^λ_WC + denom_BC^λ_BC)
                ll += log(num / denom_BC) + log(prob_nest)
            else  # Other
                prob_other = 1 / (1 + (sum(exp((dot(X[i,:], β_WC) + γ*(Z[i,j] - Z[i,J])) / λ_WC) for j in 1:3))^λ_WC + (sum(exp((dot(X[i,:], β_BC) + γ*(Z[i,j] - Z[i,J])) / λ_BC) for j in 4:7))^λ_BC)
                ll += log(prob_other)
            end
        end
        
        return -ll  # Return negative log-likelihood for minimization
    end

    function estimate_nested_logit(X, Z, y)
        n, k = size(X)
        
        # Initialize parameters
        θ_init = vcat(repeat([0.0], 2k), [1.0, 1.0], [0.0])
        
        # Optimize
        result = optimize(θ -> nested_logit_loglikelihood(θ, X, Z, y), θ_init, BFGS())
        
        return Optim.minimizer(result)
    end

    # Estimate nested logit model
    θ_nested = estimate_nested_logit(X, Z, y)

    println("\nNested Logit Estimates:")
    println(θ_nested)

    # 4. Wrap code into a function

    function run_econometrics_analysis(X, Z, y)
        println("Multinomial Logit Estimates:")
        β_mnl = estimate_mnl(X, Z, y)
        println(β_mnl)
        
        println("\nNested Logit Estimates:")
        θ_nested = estimate_nested_logit(X, Z, y)
        println(θ_nested)
    end

    # Call the function
    run_econometrics_analysis(X, Z, y)

    # 5. Unit tests

    @testset "Econometrics Analysis Tests" begin
        # Test MNL log-likelihood function
        @test mnl_loglikelihood(zeros(10), X[1:5,:], Z[1:5,:], y[1:5]) isa Float64
        
        # Test nested logit log-likelihood function
        @test nested_logit_loglikelihood(zeros(8), X[1:5,:], Z[1:5,:], y[1:5]) isa Float64
        
        # Test MNL estimation
        @test length(estimate_mnl(X[1:10,:], Z[1:10,:], y[1:10])) == size(X, 2) * (size(Z, 2) - 1) + 1
        
        # Test nested logit estimation
        @test length(estimate_nested_logit(X[1:10,:], Z[1:10,:], y[1:10])) == 2 * size(X, 2) + 3
        
        # Test run_econometrics_analysis function
        @test_nowarn run_econometrics_analysis(X[1:10,:], Z[1:10,:], y[1:10])
    end

    # Run the tests
    run_econometrics_analysis(X, Z, y)
end

# Call the allwrap function
allwrap()

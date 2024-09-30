# ECON 6343: Econometrics III 
# Student Name: Jacob Dison Wambede
# Date: 2024-08-09
# Problem Set 2: PS2-optimization-intro
#Prof. Tyler Ransom
#University of Oklahoma
#Importing required packages
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables
using CSV
using FreqTables
using Optim

# Problem 1: Basic optimization in Julia
function problem1()
    f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
    negf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2
    startval = rand(1)
    result = optimize(negf, startval, BFGS())
    println("argmin (minimizer) is ", Optim.minimizer(result)[1])
    println("min is ", Optim.minimum(result))
end

# Problem 2: OLS using Optim
function problem2()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    function ols(beta, X, y)
        ssr = (y .- X*beta)'*(y .- X*beta)
        return ssr[1]
    end

    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(),
                            Optim.Options(g_tol=1e-6, iterations=100_000))
    
    println("Problem 2 result:")
    println("OLS coefficients (Optim): ", beta_hat_ols.minimizer)
    
    # Check with built-in OLS
    bols = inv(X'*X)*X'*y
    println("OLS coefficients (manual): ", bols)
    
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    println("OLS coefficients (GLM): ", coef(bols_lm))
end

# Problem 3: Logit using Optim
function problem3()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    function logit_likelihood(beta, X, y)
        xb = X * beta
        ll = sum(y .* xb - log.(1 .+ exp.(xb)))
        return -ll  # negative because we're minimizing
    end

    beta_hat_logit = optimize(b -> logit_likelihood(b, X, y), rand(size(X,2)), LBFGS(),
                              Optim.Options(g_tol=1e-6, iterations=100_000))
    
    println("Problem 3 result:")
    println("Logit coefficients (Optim): ", beta_hat_logit.minimizer)
end

# Problem 4: Logit using GLM
function problem4()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df.white = df.race.==1
    
    logit_model = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    
    println("Problem 4 result:")
    println("Logit coefficients (GLM): ", coef(logit_model))
end

# Problem 5: Multinomial Logit using Optim
function problem5()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    function mnl_likelihood(beta, X, y)
        K, J = size(X, 2), 6
        beta_matrix = reshape(beta, K, J)
        xb = X * beta_matrix
        xb = hcat(xb, zeros(size(xb, 1)))  # Add base alternative
        prob = exp.(xb) ./ sum(exp.(xb), dims=2)
        ll = sum(log.(prob[CartesianIndex.(1:size(prob,1), y)]))
        return -ll  # negative because we're minimizing
    end

    K, J = size(X, 2), 6
    beta_start = rand(K * J)
    beta_hat_mnl = optimize(b -> mnl_likelihood(b, X, y), beta_start, LBFGS(),
                            Optim.Options(g_tol=1e-5, iterations=100_000))
    
    println("Problem 5 result:")
    println("Multinomial Logit coefficients (Optim): ", beta_hat_mnl.minimizer)
end

# Main function to run all problems
function main()
    problem1()
    problem2()
    problem3()
    problem4()
    problem5()
end

# Run the main function
main()

# Problem 7: Unit tests
using Test

@testset "Problem Set 2 Tests" begin
    @testset "Problem 1" begin
        f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
        negf(x) = -f(x)
        result = optimize(negf, [0.0], LBFGS())
        @test isapprox(result.minimizer[1], -7.38, atol=0.01)
        @test isapprox(-result.minimum, 964.3, atol=0.1)
    end

    @testset "OLS function" begin
        X = [ones(5) [1,2,3,4,5]]
        y = [2.0, 4.0, 5.0, 4.0, 5.0]
        function ols(beta, X, y)
            ssr = (y .- X*beta)'*(y .- X*beta)
            return ssr[1]
        end
        result = optimize(b -> ols(b, X, y), [0.0, 0.0], LBFGS())
        @test isapprox(result.minimizer, [2.2, 0.5], atol=0.1)
    end

    @testset "Logit likelihood function" begin
        X = [ones(5) [1,2,3,4,5]]
        y = [0, 0, 1, 1, 1]
        function logit_likelihood(beta, X, y)
            xb = X * beta
            ll = sum(y .* xb - log.(1 .+ exp.(xb)))
            return -ll
        end
        result = optimize(b -> logit_likelihood(b, X, y), [0.0, 0.0], LBFGS())
        @test length(result.minimizer) == 2
    end

    @testset "Multinomial logit likelihood function" begin
        X = [ones(5) [1,2,3,4,5]]
        y = [1, 2, 3, 2, 1]
        function mnl_likelihood(beta, X, y)
            K, J = size(X, 2), 2  # 3 choices, but we estimate 2 sets of coefficients
            beta_matrix = reshape(beta, K, J)
            xb = X * beta_matrix
            xb = hcat(xb, zeros(size(xb, 1)))
            prob = exp.(xb) ./ sum(exp.(xb), dims=2)
            ll = sum(log.(prob[CartesianIndex.(1:size(prob,1), y)]))
            return -ll
        end
        result = optimize(b -> mnl_likelihood(b, X, y), rand(4), LBFGS())
        @test length(result.minimizer) == 4
    end
end
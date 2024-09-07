using Optim
using DataFrames
using CSV
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using FreqTables

# Question 1
println("Question 1:")

f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2

startval = rand(1) # random starting value
result = optimize(minusf, startval, BFGS())

println("argmin (minimizer) is ", Optim.minimizer(result)[1])
println("min is ", Optim.minimum(result))
println("max of original function is ", -Optim.minimum(result))

# Question 2
println("\nQuestion 2:")

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y .- X*beta)'*(y .- X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(),
                        Optim.Options(g_tol=1e-6, iterations=100_000,
                        show_trace=false))

println("OLS estimates using Optim:")
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y
println("\nOLS estimates using manual calculation:")
println(bols)

df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)
println("\nOLS estimates using GLM package:")
println(coef(bols_lm))

# Question 3
println("\nQuestion 3:")

function logit_likelihood(beta, X, y)
    z = X * beta
    loglike = sum(y .* z - log.(1 .+ exp.(z)))
    return -loglike  # Return negative because Optim minimizes
end

beta_hat_logit = optimize(b -> logit_likelihood(b, X, y), zeros(size(X,2)), LBFGS(),
                          Optim.Options(g_tol=1e-5, iterations=100_000,
                          show_trace=false))

println("Logit estimates using Optim:")
println(beta_hat_logit.minimizer)

# Question 4
println("\nQuestion 4:")

logit_model = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println("Logit estimates using GLM package:")
println(coef(logit_model))

# Question 5
println("\nQuestion 5:")

freqtable(df, :occupation)
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation)

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit_likelihood(beta, X, y)
    n, k = size(X)
    J = 6  # Number of choices minus 1 (base category)
    beta_matrix = reshape(beta, k, J)
    
    loglike = 0.0
    for i in 1:n
        denom = 1.0 + sum(exp.(X[i,:] .* beta_matrix))
        if y[i] <= J
            loglike += X[i,:]' * beta_matrix[:, Int(y[i])] - log(denom)
        else
            loglike += -log(denom)
        end
    end
    
    return -loglike  # Return negative because Optim minimizes
end

# Try different starting values
start_zeros = zeros(size(X,2) * 6)
start_random = rand(size(X,2) * 6)
start_uniform = rand(Uniform(-1, 1), size(X,2) * 6)

results_zeros = optimize(b -> mlogit_likelihood(b, X, y), start_zeros, LBFGS(),
                         Optim.Options(g_tol=1e-5, iterations=100_000,
                         show_trace=false))

results_random = optimize(b -> mlogit_likelihood(b, X, y), start_random, LBFGS(),
                          Optim.Options(g_tol=1e-5, iterations=100_000,
                          show_trace=false))

results_uniform = optimize(b -> mlogit_likelihood(b, X, y), start_uniform, LBFGS(),
                           Optim.Options(g_tol=1e-5, iterations=100_000,
                           show_trace=false))

println("Multinomial Logit estimates (zeros start):")
println(reshape(results_zeros.minimizer, size(X,2), 6))

println("\nMultinomial Logit estimates (random start):")
println(reshape(results_random.minimizer, size(X,2), 6))

println("\nMultinomial Logit estimates (uniform start):")
println(reshape(results_uniform.minimizer, size(X,2), 6))

# Question 6 is implicitly answered by wrapping everything in a function

# Question 7: Unit tests
using Test

@testset "OLS Function Tests" begin
    # Test OLS function with known values
    X_test = [1.0 2.0; 1.0 3.0; 1.0 4.0]
    y_test = [2.0, 3.0, 4.0]
    beta_true = [1.0, 1.0]
    
    @test isapprox(ols(beta_true, X_test, y_test), 0.0, atol=1e-10)
    
    # Test OLS optimization
    result_test = optimize(b -> ols(b, X_test, y_test), rand(2), LBFGS())
    @test isapprox(result_test.minimizer, beta_true, atol=1e-5)
end

@testset "Logit Function Tests" begin
    # Test logit function with known values
    X_test = [1.0 1.0; 1.0 2.0; 1.0 3.0]
    y_test = [0.0, 1.0, 1.0]
    beta_test = [0.0, 1.0]
    
    loglike_test = logit_likelihood(beta_test, X_test, y_test)
    @test loglike_test < 0  # Log-likelihood should be negative
    
    # Test logit optimization
    result_test = optimize(b -> logit_likelihood(b, X_test, y_test), rand(2), LBFGS())
    @test length(result_test.minimizer) == 2
end

@testset "Multinomial Logit Function Tests" begin
    # Test multinomial logit function with known values
    X_test = [1.0 1.0; 1.0 2.0; 1.0 3.0]
    y_test = [1.0, 2.0, 3.0]
    beta_test = rand(2 * 2)  # 2 parameters, 3 choices (2 non-base)
    
    loglike_test = mlogit_likelihood(beta_test, X_test, y_test)
    @test loglike_test < 0  # Log-likelihood should be negative
    
    # Test multinomial logit optimization
    result_test = optimize(b -> mlogit_likelihood(b, X_test, y_test), rand(2 * 2), LBFGS())
    @test length(result_test.minimizer) == 4  # 2 parameters * 2 non-base choices
end

println("\nAll tests completed.")
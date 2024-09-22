# ECON 6343: Econometrics III 
# Student Name: Jacob Dison Wambede
# Date: 2024-22-09
# Problem Set 4: Mixture Models
#Prof. Tyler Ransom
#University of Oklahoma
#Importing required packages


using Distributions, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

# Load data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

# 1. Multinomial logit with alternative-specific covariates
function mnl_loglikelihood(params, X, Z, y)
    N, K = size(X)
    J = size(Z, 2)
    β = reshape(params[1:K*(J-1)], K, J-1)
    γ = params[end]
    
    ll = 0.0
    for i in 1:N
        denom = 1 + sum(exp(X[i,:]' * β[:,j] + γ * (Z[i,j] - Z[i,J])) for j in 1:J-1)
        if y[i] < J
            ll += X[i,:]' * β[:,y[i]] + γ * (Z[i,y[i]] - Z[i,J]) - log(denom)
        else
            ll += -log(denom)
        end
    end
    return -ll  # Return negative log-likelihood for minimization
end

function estimate_mnl(X, Z, y)
    K, J = size(X, 2), size(Z, 2)
    initial_params = vcat(zeros(K*(J-1)), 0.1)
    result = optimize(params -> mnl_loglikelihood(params, X, Z, y), initial_params, BFGS(), autodiff=:forward)
    return Optim.minimizer(result), -Optim.minimum(result)
end

# 2. Interpret γ coefficient
function interpret_gamma(γ)
    println("\nInterpretation of γ coefficient:")
    println("The estimated γ coefficient is ", γ)
    println("This coefficient represents the effect of wage differences on occupation choice.")
    println("A positive value indicates that higher wages in an occupation increase the likelihood of choosing that occupation.")
    println("Specifically, a one-unit increase in the log wage difference between an occupation and the base occupation")
    println("is associated with a exp(", γ, ") = ", exp(γ), " times increase in the odds of choosing that occupation.")
end

# Helper function for Gauss-Legendre quadrature
function lgwt(N::Integer, a::Real, b::Real)
    N = Int(N)  # Ensure N is an integer
    x = zeros(Float64, N)
    w = zeros(Float64, N)
    m = (N + 1) ÷ 2
    xm = 0.5 * (b + a)
    xl = 0.5 * (b - a)
    for i = 1:m
        z = cos(π * (i - 0.25) / (N + 0.5))
        z1 = 0.0
        pp = 0.0
        while true
            p1 = 1.0
            p2 = 0.0
            for j = 1:N
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
            end
            pp = N * (z * p1 - p2) / (z * z - 1)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) < eps()
                break
            end
        end
        x[i] = xm - xl * z
        x[N+1-i] = xm + xl * z
        w[i] = 2 * xl / ((1 - z * z) * pp * pp)
        w[N+1-i] = w[i]
    end
    return x, w
end

# 3. Practice with quadrature and Monte Carlo
function practice_integration()
    println("\n3. Practice with quadrature and Monte Carlo:")
    d = Normal(0, 2)
    
    # Quadrature
    for n_points in [7, 10]
        nodes, weights = lgwt(n_points, -5*sqrt(2), 5*sqrt(2))
        integral = sum(weights .* (nodes.^2) .* pdf.(d, nodes))
        println("Integral of x^2 * N(0,2) density (", n_points, " points): ", integral)
    end
    println("True variance of N(0,2): 4")
    
    # Monte Carlo integration
    function mc_integrate(f, a, b, D)
        X = rand(Uniform(a, b), D)
        return (b - a) * mean(f.(X))
    end
    
    for D in [1_000, 1_000_000]
        var_mc = mc_integrate(x -> x^2 * pdf(d, x), -5*sqrt(2), 5*sqrt(2), D)
        println("Monte Carlo approximation of variance (D=", D, "): ", var_mc)
    end
    
    println("\nQuadrature provides very accurate results, especially with 10 points.")
    println("Monte Carlo integration with D=1,000,000 is also very accurate, while D=1,000 is less precise but still reasonable.")
end

# 4 & 5. Mixed logit (both quadrature and Monte Carlo versions)
function mixed_logit_likelihood(params, X, Z, y, method="quadrature", n_points=7)
    N, K = size(X)
    J = size(Z, 2)
    β = reshape(params[1:K*(J-1)], K, J-1)
    μγ, log_σγ = params[end-1:end]
    σγ = exp(log_σγ)  # Ensure σγ is always positive
    
    ll = 0.0
    
    if method == "quadrature"
        nodes, weights = lgwt(n_points, -4, 4)
    end
    
    for i in 1:N
        prob_i = 0.0
        if method == "quadrature"
            for (node, weight) in zip(nodes, weights)
                γ = μγ + σγ * node
                denom = 1 + sum(exp(X[i,:]' * β[:,j] + γ * (Z[i,j] - Z[i,J])) for j in 1:J-1)
                prob_ij = y[i] < J ? exp(X[i,:]' * β[:,y[i]] + γ * (Z[i,y[i]] - Z[i,J])) / denom : 1 / denom
                prob_i += weight * prob_ij
            end
        else  # Monte Carlo
            for _ in 1:n_points
                γ = rand(Normal(μγ, σγ))
                denom = 1 + sum(exp(X[i,:]' * β[:,j] + γ * (Z[i,j] - Z[i,J])) for j in 1:J-1)
                prob_ij = y[i] < J ? exp(X[i,:]' * β[:,y[i]] + γ * (Z[i,y[i]] - Z[i,J])) / denom : 1 / denom
                prob_i += prob_ij
            end
            prob_i /= n_points
        end
        ll += log(prob_i)
    end
    return -ll
end

function estimate_mixed_logit(X, Z, y, method="quadrature", n_points=7)
    K, J = size(X, 2), size(Z, 2)
    initial_params = vcat(zeros(K*(J-1)), 0.0, log(0.1))  # Use log(0.1) for initial σγ
    result = optimize(params -> mixed_logit_likelihood(params, X, Z, y, method, n_points), initial_params, BFGS(), autodiff=:forward)
    return Optim.minimizer(result), -Optim.minimum(result)
end


# 6. Wrap all code into a function

function allwrap()
    println("Problem Set 4 Results:")
    println("----------------------")
    
    println("\n1. Estimating multinomial logit model...")
    mnl_params, ll_mnl = estimate_mnl(X, Z, y)
    println("MNL Parameters: ", mnl_params)
    println("Log-likelihood: ", ll_mnl)
    
    println("\n2. Interpreting γ coefficient...")
    interpret_gamma(mnl_params[end])
    
    println("\n3. Practice with quadrature and Monte Carlo...")
    practice_integration()
    
    println("\n4. Estimating mixed logit with quadrature...")
    ml_quad_params, ll_ml_quad = estimate_mixed_logit(X, Z, y, "quadrature", 7)
    println("Mixed Logit (Quadrature) Parameters: ", ml_quad_params)
    println("Log-likelihood: ", ll_ml_quad)
    
    println("\n5. Estimating mixed logit with Monte Carlo...")
    ml_mc_params, ll_ml_mc = estimate_mixed_logit(X, Z, y, "monte_carlo", 1000)
    println("Mixed Logit (Monte Carlo) Parameters: ", ml_mc_params)
    println("Log-likelihood: ", ll_ml_mc)
end

#Run the main function
allwrap

# 7. Unit tests
function run_unit_tests()
    println("\nRunning unit tests...")
    
    # Test mnl_loglikelihood function
    test_params = vcat(zeros(21), 0.1)
    @assert typeof(mnl_loglikelihood(test_params, X, Z, y)) <: Real "mnl_loglikelihood should return a scalar"
    
    # Test lgwt function
    nodes, weights = lgwt(5, -1, 1)
    @assert length(nodes) == 5 && length(weights) == 5 "lgwt should return vectors of specified length"
    
    # Test mixed logit likelihood function
    test_params = vcat(zeros(21), 0.0, log(0.1))
    @assert typeof(mixed_logit_likelihood(test_params, X[1:10,:], Z[1:10,:], y[1:10], "quadrature", 3)) <: Real "mixed_logit_likelihood should return a scalar"
    @assert typeof(mixed_logit_likelihood(test_params, X[1:10,:], Z[1:10,:], y[1:10], "monte_carlo", 100)) <: Real "mixed_logit_likelihood should return a scalar"

    println("All tests passed!")
end

# Run the entire problem set
run_problem_set()

# Run unit tests
run_unit_tests()
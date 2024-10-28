using DataFrames
using CSV
using HTTP
using Statistics
using LinearAlgebra
using Optim
using GLM
using Random
using Distributions 
using MultivariateStats
using Test
using Dates

# Load data from URL
function load_nlsy_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS8-factor/nlsy.csv"
    response = HTTP.get(url)
    data = CSV.read(IOBuffer(String(response.body)), DataFrame)
    return data
end

# Load data
data = load_nlsy_data()

# Show first few rows and summary
println("\nFirst few rows of data:")
println(first(data, 5))

println("\nDataset dimensions:")
println(size(data))

# Print start time
println("Start date/time: ", Dates.now())

#==========================================================
Question 1: Basic Linear Regression
==========================================================#
function basic_regression(df)
    println("\nQuestion 1: Basic Linear Regression")
    model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
    println(model)
    return model
end

#==========================================================
Question 2: ASVAB Correlation
==========================================================#
function compute_asvab_correlation(df)
    println("\nQuestion 2: ASVAB Correlation Matrix")
    asvab_cols = [:asvabMK, :asvabWK, :asvabAR, :asvabCS, :asvabNO, :asvabPC]
    asvab_mat = Matrix(df[:, asvab_cols])
    corr_mat = cor(asvab_mat)
    println("Correlation matrix of ASVAB scores:")
    display(round.(corr_mat, digits=3))
    return corr_mat
end

#==========================================================
Question 3: Regression with ASVAB
==========================================================#
function regression_with_asvab(df)
    println("\nQuestion 3: Regression with ASVAB variables")
    model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
                       asvabMK + asvabWK + asvabAR + asvabCS + asvabNO + asvabPC), df)
    println(model)
    return model
end

#==========================================================
Question 4: PCA Analysis
==========================================================#
function pca_regression(df)
    println("\nQuestion 4: PCA Regression")
    
    # Prepare ASVAB matrix
    asvab_cols = [:asvabMK, :asvabWK, :asvabAR, :asvabCS, :asvabNO, :asvabPC]
    asvab_mat = Matrix(df[:, asvab_cols])'  # Transpose for PCA
    
    # Fit PCA
    M = fit(PCA, asvab_mat; maxoutdim=1)
    
    # Get first principal component
    asvab_pca = vec(MultivariateStats.transform(M, asvab_mat)')
    
    # Add PCA component to dataframe
    df_pca = copy(df)
    df_pca.asvab_pc1 = asvab_pca
    
    # Run regression with PCA component
    model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvab_pc1), df_pca)
    println(model)
    return model, M
end

#==========================================================
Question 5: Factor Analysis
==========================================================#
function fa_regression(df)
    println("\nQuestion 5: Factor Analysis Regression")
    
    # Prepare ASVAB matrix
    asvab_cols = [:asvabMK, :asvabWK, :asvabAR, :asvabCS, :asvabNO, :asvabPC]
    asvab_mat = Matrix(df[:, asvab_cols])'  # Transpose for FA
    
    # Fit Factor Analysis
    M = fit(FactorAnalysis, asvab_mat; maxoutdim=1)
    
    # Get factor scores
    asvab_fa = vec(MultivariateStats.transform(M, asvab_mat)')
    
    # Add FA component to dataframe
    df_fa = copy(df)
    df_fa.asvab_factor = asvab_fa
    
    # Run regression with FA component
    model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvab_factor), df_fa)
    println(model)
    return model, M
end

#==========================================================
Question 6: Maximum Likelihood Estimation
==========================================================#
function gausslegendre(n::Integer)
    # Calculate Gauss-Legendre quadrature points and weights
    i = 1:n-1
    β = i./sqrt.(4*i.^2 .- 1)
    J = SymTridiagonal(zeros(n), β)
    nodes, v = eigen(J)
    weights = 2*v[1,:].^2
    return nodes, weights
end

function loglikelihood(params, df, x_vars, quad_nodes, quad_weights)
    # Unpack parameters
    α₀ = params[1:6]         # intercepts for ASVAB equations
    α₁ = params[7:12]        # black coefficients
    α₂ = params[13:18]       # hispanic coefficients
    α₃ = params[19:24]       # female coefficients
    γ = params[25:30]        # factor loadings
    σ_asvab = exp.(params[31:36])  # ASVAB error standard deviations
    β = params[37:43]        # wage equation coefficients
    δ = params[44]           # wage equation factor loading
    σ_w = exp(params[45])    # wage equation error standard deviation
    
    # Get data matrices
    X_m = hcat(ones(nrow(df)), df.black, df.hispanic, df.female)
    X = hcat(ones(nrow(df)), Matrix(df[:, x_vars]))
    
    # Initialize log-likelihood
    loglik = 0.0
    
    # ASVAB column names
    asvab_cols = [:asvabMK, :asvabWK, :asvabAR, :asvabCS, :asvabNO, :asvabPC]
    
    # Loop over observations
    for i in 1:nrow(df)
        # Initialize integral approximation
        integral = 0.0
        
        # Gauss-Legendre quadrature
        for (node, weight) in zip(quad_nodes, quad_weights)
            # Measurement equations likelihood
            meas_like = 1.0
            for j in 1:6
                asvab_pred = X_m[i,:]'*[α₀[j]; α₁[j]; α₂[j]; α₃[j]] + γ[j]*node
                meas_like *= pdf(Normal(0,σ_asvab[j]), 
                               df[i, asvab_cols[j]] - asvab_pred)
            end
            
            # Wage equation likelihood
            wage_pred = X[i,:]'*β + δ*node
            wage_like = pdf(Normal(0,σ_w), df.logwage[i] - wage_pred)
            
            # Multiply by standard normal density of factor
            integral += weight * meas_like * wage_like * pdf(Normal(0,1), node)
        end
        
        loglik += log(integral)
    end
    
    return -loglik  # negative because we're minimizing
end

function estimate_mle(df)
    println("\nQuestion 6: Maximum Likelihood Estimation")
    
    # Setup quadrature
    n_quad = 7
    quad_nodes, quad_weights = gausslegendre(n_quad)
    
    # Setup variables
    x_vars = [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]
    
    # Initial parameter values
    n_params = 45  # Total number of parameters
    initial_params = zeros(n_params)
    initial_params[31:36] .= log(1.0)  # log standard deviations
    initial_params[45] = log(1.0)      # log wage equation std dev
    
    # Optimize with error handling
    try
        result = optimize(params -> loglikelihood(params, df, x_vars, quad_nodes, quad_weights),
                         initial_params,
                         LBFGS(),
                         Optim.Options(iterations=1000, show_trace=true))
        
        println("\nOptimization complete:")
        println("Convergence: ", Optim.converged(result))
        println("Final log-likelihood: ", -result.minimum)
        
        return result
    catch e
        println("Error in MLE estimation: ", e)
        return nothing
    end
end

#==========================================================
Question 7: Unit Tests
==========================================================#
function run_tests()
    println("\nQuestion 7: Running Unit Tests")
    
    @testset "Problem Set 8 Tests" begin
        # Test data loading
        @testset "Data Loading" begin
            df = load_data()
            @test nrow(df) > 0
            @test :log_wage in propertynames(df)
            @test all([:asvabMK, :asvabWK, :asvabAR, :asvabCS, :asvabNO, :asvabPC] .∈ Ref(propertynames(df)))
        end

        # Test basic regression
        @testset "Basic Regression" begin
            df = load_data()
            model = basic_regression(df)
            @test model isa GLM.LinearModel
            @test coef(model) isa Vector{Float64}
            @test length(coef(model)) == 7  # intercept + 6 variables
        end

        # Test ASVAB correlation
        @testset "ASVAB Correlation" begin
            df = load_data()
            corr_mat = compute_asvab_correlation(df)
            @test size(corr_mat) == (6,6)
            @test all(diag(corr_mat) .≈ 1.0)
            @test issymmetric(corr_mat)
        end

        # Test PCA
        @testset "PCA Analysis" begin
            df = load_data()
            model, M = pca_regression(df)
            @test model isa GLM.LinearModel
            @test M isa PCA
            @test size(projection(M), 2) == 1
        end

        # Test Factor Analysis
        @testset "Factor Analysis" begin
            df = load_data()
            model, M = fa_regression(df)
            @test model isa GLM.LinearModel
            @test M isa FactorAnalysis
            @test size(projection(M), 2) == 1
        end

        # Test Gaussian Quadrature
        @testset "Gaussian Quadrature" begin
            nodes, weights = gausslegendre(7)
            @test length(nodes) == 7
            @test length(weights) == 7
            @test sum(weights) ≈ 2.0 rtol=1e-10
        end
    end
end

#==========================================================
Main Execution
==========================================================#
function main()
    try
        # Load data
        df = load_data()
        
        # Run all analyses
        basic_regression(df)
        compute_asvab_correlation(df)
        regression_with_asvab(df)
        pca_regression(df)
        fa_regression(df)
        estimate_mle(df)
        run_tests()
    catch e
        println("Error in main execution: ", e)
        rethrow(e)
    finally
        println("\nEnd date/time: ", Dates.now())
    end
end

# Execute main function
main()
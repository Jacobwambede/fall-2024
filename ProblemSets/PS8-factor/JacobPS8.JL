using HTTP, CSV, DataFrames

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

# Load required packages
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV, MultivariateStats
using Distributions


# Question 1: Basic linear regression model
function basic_regression(data)
    # Specify model: log(wage) = β₀ + β₁black + β₂hispanic + β₃female + β₄school + β₅gradHS + β₆grad4yr + ε
    formula = @formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr)
    
    # Estimate model
    model = lm(formula, data)
    
    return model
end

# Load data
println("Loading NLSY data...")
data = load_nlsy_data()

# Run regression
println("\nQuestion 1: Basic Wage Regression Results")
println("==========================================")
model1 = basic_regression(data)

# Print detailed results
println("\nRegression Summary:")
println(model1)

# Print coefficient table
println("\nCoefficient Estimates:")
coef_table = coeftable(model1)
println(coef_table)

# Print model fit statistics
r2 = r²(model1)
n = nobs(model1)
p = length(coef(model1))
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)

println("\nModel Fit Statistics:")
println("R-squared: ", round(r2, digits=4))
println("Adjusted R-squared: ", round(adj_r2, digits=4))
println("Number of observations: ", n)

# Compute correlations between ASVAB variables
function compute_asvab_correlations(data)
    # Get ASVAB column names
    asvab_cols = [col for col in names(data) if startswith(String(col), "asvab")]
    
    # Extract ASVAB matrix
    asvab_matrix = Matrix(data[:, asvab_cols])
    
    # Compute correlation matrix
    correlation_matrix = cor(asvab_matrix)
    
    return correlation_matrix, asvab_cols
end

# Calculate and display correlations
cor_matrix, asvab_vars = compute_asvab_correlations(data)

# Print results
println("\nASVAB Variable Names:")
println(asvab_vars)

println("\nCorrelation Matrix:")
# Print with variable names and formatted numbers
println("      ", join(lpad.(asvab_vars, 8), " "))
for i in 1:length(asvab_vars)
    print(lpad(asvab_vars[i], 6), " ")
    println(join(lpad.(round.(cor_matrix[i,:], digits=3), 8), " "))
end
# Question 3: Full regression with all ASVAB variables
function full_regression(data)
    # Get ASVAB column names
    asvab_cols = [col for col in names(data) if startswith(String(col), "asvab")]
    
    # Create the formula term with base variables and all ASVAB scores
    term = Term(:logwage) ~ sum(Term.([:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr])) + sum(Term.(Symbol.(asvab_cols)))
    
    # Estimate model
    model = lm(term, data)
    return model
end

# Run regression and display results
println("\nQuestion 3: Full Regression with All ASVAB Variables")
println("=================================================")
model3 = full_regression(data)
println("\nRegression Summary:")
println(model3)

# Print coefficient table
println("\nCoefficient Estimates:")
coeftab = coeftable(model3)
println(coeftab)

# Print model fit statistics
r2 = r²(model3)
adj_r2 = adjr²(model3)
println("\nR-squared: ", round(r2, digits=4))
println("Adjusted R-squared: ", round(adj_r2, digits=4))

# Question 4: PCA regression
function pca_regression(data)
    # Get ASVAB variables
    asvab_cols = [col for col in names(data) if startswith(String(col), "asvab")]
    asvab_matrix = Matrix(data[:, asvab_cols])'  # Transpose for MultivariateStats
    
    # Standardize ASVAB data
    asvab_std = (asvab_matrix .- mean(asvab_matrix, dims=2)) ./ std(asvab_matrix, dims=2)
    
    # Fit PCA
    M = fit(PCA, asvab_std; maxoutdim=1)
    
    # Get first principal component
    asvab_pca = vec(MultivariateStats.transform(M, asvab_std)')
    
    # Create DataFrame with PCA scores
    data_with_pca = copy(data)
    data_with_pca.asvab_pca = asvab_pca
    
    # Estimate model with first principal component
    formula = @formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvab_pca)
    model = lm(formula, data_with_pca)
    
    # Get explained variance ratio
    explained_var = principalratio(M)
    
    return model, M, explained_var
end

# Question 5: Factor Analysis regression
function fa_regression(data)
    # Get ASVAB variables
    asvab_cols = [col for col in names(data) if startswith(String(col), "asvab")]
    asvab_matrix = Matrix(data[:, asvab_cols])'  # Transpose for MultivariateStats
    
    # Standardize ASVAB data
    asvab_std = (asvab_matrix .- mean(asvab_matrix, dims=2)) ./ std(asvab_matrix, dims=2)
    
    # Fit Factor Analysis
    M = fit(FactorAnalysis, asvab_std; maxoutdim=1)
    
    # Get factor scores
    asvab_fa = vec(MultivariateStats.transform(M, asvab_std)')
    
    # Create DataFrame with FA scores
    data_with_fa = copy(data)
    data_with_fa.asvab_fa = asvab_fa
    
    # Estimate model with factor scores
    formula = @formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvab_fa)
    model = lm(formula, data_with_fa)
    
    # Get factor loadings
    loadings = M.W
    
    return model, M, loadings
end

# Run analyses
println("\nQuestion 3: Full Regression with All ASVAB Variables")
println("=================================================")
model3 = full_regression(data)
println(model3)

println("\nQuestion 4: PCA Regression")
println("=========================")
model4, pca_fit, explained_var = pca_regression(data)
println(model4)
println("\nProportion of variance explained by first PC: ", round(explained_var[1], digits=4))

println("\nQuestion 5: Factor Analysis Regression")
println("====================================")
model5, fa_fit, loadings = fa_regression(data)
println(model5)
println("\nFactor loadings:")
asvab_cols = [col for col in names(data) if startswith(String(col), "asvab")]
for (var, loading) in zip(asvab_cols, loadings)
    println(var, ": ", round(loading, digits=4))
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
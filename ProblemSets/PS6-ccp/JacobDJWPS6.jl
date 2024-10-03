# ECON 6343: Econometrics III 
# Student Name: Jacob Dison Wambede
# Date: 2024-06-10
# Problem Set : PS6-ccp
#Prof. Tyler Ransom
#University of Oklahoma



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

# Define the prepare_dynamic_data function
function prepare_dynamic_data(df::DataFrame)
    # Placeholder implementation for prepare_dynamic_data
    # This function should return the necessary data structures
    Y = df.Y
    X = hcat(df.X1, df.X2, df.X3)
    Z = hcat(df.Z1, df.Z2, df.Z3)
    B = df.B
    N = size(df, 1)
    T = 10  # Example value for T
    Xstate = hcat(df.Xstate1, df.Xstate2)
    Zstate = hcat(df.Zstate1, df.Zstate2)
    return Y, X, Z, B, N, T, Xstate, Zstate
end

# Define other necessary functions
function create_state_dataframe(xval, zval, xbin, zbin)
    # Placeholder implementation for create_state_dataframe
    return DataFrame(xval=xval, zval=zval, xbin=xbin, zbin=zbin)
end

function compute_future_value(state_df, θ̂_flex, Xstate, Zstate, xtran, xbin, zbin, T, discount_factor)
    # Placeholder implementation for compute_future_value
    return rand(size(state_df, 1))
end

function estimate_structural_parameters(df_long::DataFrame)
    # Corrected implementation for estimate_structural_parameters
    model = GLM.glm(@formula(Y ~ Odometer + Branded), df_long, Binomial())
    return model
end

function custom_logit_estimation(X_custom::Matrix{Float64}, y_custom::Vector{Float64}, offset_custom::Vector{Float64})
    # Corrected implementation for custom_logit_estimation
    df_custom = DataFrame(X_custom, :auto)
    df_custom.y_custom = y_custom
    df_custom.offset_custom = offset_custom
    model = GLM.glm(@formula(y_custom ~ x1 + x2 + x3), df_custom, Binomial(), offset=df_custom.offset_custom)
    return coef(model)
end

# Define the run_analysis function
function run_analysis()
    df = DataFrame(X1=rand(100), X2=rand(100), X3=rand(100), Z1=rand(100), Z2=rand(100), Z3=rand(100), B=rand(100), Y=rand(100), Xstate1=rand(100), Xstate2=rand(100), Zstate1=rand(100), Zstate2=rand(100), Odometer=rand(100), Branded=rand(100), fv=rand(100))
    df_long = DataFrame(Odometer=rand(100), Branded=rand(100), Y=rand(100), fv=rand(100))
    xval = rand(100)
    zval = rand(100)
    xbin = rand(100)
    zbin = rand(100)
    xtran = rand(100, 100)
    θ̂_flex = rand(10)
    
    state_df = create_state_dataframe(xval, zval, xbin, zbin)
    
    Y, X, Z, B, N, T, Xstate, Zstate = prepare_dynamic_data(df)
    
    fvt1 = compute_future_value(state_df, θ̂_flex, Xstate, Zstate, xtran, xbin, zbin, T, 0.9)
    
    # Estimate structural parameters
    θ̂_ccp_glm = estimate_structural_parameters(df_long)
    println("Structural Parameters (GLM with offset):")
    println(θ̂_ccp_glm)
    
    # Custom logit estimation
    X_custom = hcat(ones(size(df_long, 1)), df_long.Odometer, df_long.Branded)
    y_custom = convert(Vector{Float64}, df_long.Y)
    offset_custom = df_long.fv
    θ̂_ccp_custom = custom_logit_estimation(X_custom, y_custom, offset_custom)
    println("Structural Parameters (Custom Logit):")
    println(θ̂_ccp_custom)
end

# Run the analysis
@time run_analysis()

# Unit tests
@testset "Model Tests" begin
    @testset "estimate_structural_parameters" begin
        test_df_long = DataFrame(
            Y = [0, 1, 1, 0],
            Odometer = [100, 150, 200, 250],
            Branded = [0, 0, 1, 1],
            fv = [0.1, 0.2, 0.3, 0.4]
        )
        model = estimate_structural_parameters(test_df_long)
        @test isa(model, StatsModels.TableRegressionModel)
        @test length(coef(model)) == 3  # Intercept, Odometer, Branded
    end

    @testset "custom_logit_estimation" begin
        X = hcat(ones(4), [100, 150, 200, 250], [0, 0, 1, 1])
        y = convert(Vector{Float64}, [0, 1, 1, 0])
        offset = [0.1, 0.2, 0.3, 0.4]
        θ̂ = custom_logit_estimation(X, y, offset)
        @test length(θ̂) == 4  # Intercept, x1, x2, x3
    end
end
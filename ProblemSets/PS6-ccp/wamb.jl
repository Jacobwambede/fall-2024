
using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM
using Test

# Assuming create_grids.jl is in the same directory
include("create_grids.jl")

#:::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: read in data and reshape to long format
#:::::::::::::::::::::::::::::::::::::::::::::::::::::
function reshape_variable(df, prefix, num_periods, new_name=nothing)
    cols = [Symbol(prefix, i) for i in 1:num_periods]
    df_subset = select(df, :bus_id, cols...)
    df_long = stack(df_subset, cols, variable_name=:time, value_name=isnothing(new_name) ? string(prefix) : string(new_name))
    df_long.time = parse.(Int, replace.(string.(df_long.time), string(prefix) => ""))
    sort!(df_long, [:bus_id, :time])
    return df_long
end

function import_and_prepare_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df.bus_id = 1:size(df, 1)
    
    dfy_long = reshape_variable(df, :Y, 20)
    dfx_long = reshape_variable(df, :Odo, 20, :Odometer)
    dfxs_long = reshape_variable(df, :Xst, 20)
    
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    df_long = leftjoin(df_long, dfxs_long, on = [:bus_id, :time])
    df_long = leftjoin(df_long, select(df, :bus_id, :RouteUsage, :Branded), on = :bus_id)
    sort!(df_long, [:bus_id, :time])
    
    return df, df_long
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: estimate flexible logit (CCP parameters)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function estimate_flexible_logit(df_long)
    return glm(@formula(Y ~ Odometer * Odometer * RouteUsage * RouteUsage * Branded * time * time), 
               df_long, Binomial(), LogitLink())
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3: create state bin data frame
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function create_state_bin_dataframe()
    zval, zbin, xval, xbin, xtran = create_grids()
    return DataFrame(
        Odometer = kron(ones(zbin), xval),
        RouteUsage = kron(zval, ones(xbin)),
        time = zeros(size(xtran,1)),
        Branded = zeros(size(xtran,1))
    ), xtran, xbin, zbin
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4: compute FV using CCPs
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function compute_future_value(b1, β, df, xtran, xbin, zbin, N, T, Xstate, Zstate, B)
    FV1 = zeros(xbin*zbin, 2, T+1)
    FVT1 = zeros(N, T)
    
    for t in 2:T, s in 0:1
        @with(df, :time .= t)
        @with(df, :Branded .= s)
        p0 = 1 .- convert(Array{Float64}, predict(b1, df))
        FV1[:, s+1, t] = -β .* log.(p0)
    end
    
    for i in 1:N
        row0 = (Zstate[i]-1)*xbin + 1
        for t in 1:T
            row1 = row0 + Xstate[i,t] - 1
            FVT1[i,t] = (xtran[row1,:] .- xtran[row0,:]) ⋅ FV1[row0:row0+xbin-1, B[i]+1, t+1]
        end
    end
    
    return FVT1'[:]
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 5: estimate structural parameters
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function estimate_structural_parameters(df_long)
    return glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink(), offset=df_long.fv)
end

# You can also do this via Optim, and you'll get the exact same answer
function likebus_ccp(α, Y, X, B, FV, N, T)
    like = 0
    for i in 1:N, t in 1:T
        v1 = α[1] + α[2]*X[i,t] + α[3]*B[i] + FV[i,t]
        dem = 1 + exp(v1)
        like -= ((Y[i,t]==1)*v1) - log(dem)
    end
    return like
end

function estimate_parameters_optim(Y, X, B, FV, N, T)
    θ_true = [2; -.15; 1]
    θ̂_optim = optimize(a -> likebus_ccp(a, Y, X, B, FV, N, T), θ_true, LBFGS(), 
                       Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
    return θ̂_optim.minimizer
end

# Wrap in one big function
function main()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df, df_long = import_and_prepare_data(url)
    
    θ̂_flex = estimate_flexible_logit(df_long)
    println("Flexible Logit Estimates:")
    println(θ̂_flex)
    
    df_state, xtran, xbin, zbin = create_state_bin_dataframe()
    
    Y = Matrix(df[:, [Symbol("Y$i") for i in 1:20]])
    X = Matrix(df[:, [Symbol("Odo$i") for i in 1:20]])
    B = Vector(df[:, :Branded])
    Xstate = Matrix(df[:, [Symbol("Xst$i") for i in 1:20]])
    Zstate = Vector(df[:, :Zst])
    N, T = size(Zstate, 1), size(Xstate, 2)
    
    fvt1 = compute_future_value(θ̂_flex, 0.9, df_state, xtran, xbin, zbin, N, T, Xstate, Zstate, B)
    df_long.fv = fvt1
    
    θ̂_ccp_glm = @time estimate_structural_parameters(df_long)
    println("\nStructural Parameter Estimates (GLM):")
    println(θ̂_ccp_glm)
    
    FV = reshape(df_long.fv, T, N)'
    θ̂_ccp = @time estimate_parameters_optim(Y, X, B, FV, N, T)
    println("\nStructural Parameter Estimates (Optim):")
    println(θ̂_ccp)
end

# Run the main function (and time it)
@time main()


# Unit tests
println("")
println("")
println("=====================================================================================================")
println("Unit tests")
println("=====================================================================================================")
println("")
println("")
# Unit tests
@testset "Bus Data Analysis Tests" begin
    @testset "Data Import and Preparation" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
        df, df_long = import_and_prepare_data(url)
        
        @test size(df, 2) == 85  # Assuming 85 columns in original data
        @test size(df_long, 2) == 7  # bus_id, time, Y, Odometer, Xst, RouteUsage, Branded
        @test maximum(df_long.time) == 20
        @test minimum(df_long.time) == 1
    end
    
    @testset "Flexible Logit Estimation" begin
        # Create a small mock dataset
        mock_df_long = DataFrame(
            Y = rand(0:1, 100),
            Odometer = rand(1000:5000, 100),
            RouteUsage = rand(1:5, 100),
            Branded = rand(0:1, 100),
            time = repeat(1:20, 5)
        )
        
        θ̂_flex = estimate_flexible_logit(mock_df_long)
        
        @test isa(θ̂_flex, GLM.GeneralizedLinearModel)
        @test length(coef(θ̂_flex)) > 0
    end
    
    @testset "State Bin DataFrame Creation" begin
        df_state, xtran, xbin, zbin = create_state_bin_dataframe()
        
        @test size(df_state, 2) == 4  # Odometer, RouteUsage, time, Branded
        @test all(df_state.time .== 0)
        @test all(df_state.Branded .== 0)
    end
    
    @testset "Future Value Computation" begin
        # This test would require more setup, so we'll just check the function signature
        @test methods(compute_future_value).ms[1].nargs == 13
    end
    
    @testset "Structural Parameter Estimation" begin
        mock_df_long = DataFrame(
            Y = rand(0:1, 100),
            Odometer = rand(1000:5000, 100),
            Branded = rand(0:1, 100),
            fv = randn(100)
        )
        
        θ̂_ccp_glm = estimate_structural_parameters(mock_df_long)
        
        @test isa(θ̂_ccp_glm, GLM.GeneralizedLinearModel)
        @test length(coef(θ̂_ccp_glm)) == 3  # Intercept, Odometer, Branded
    end
    
    @testset "Likelihood Function" begin
        N, T = 5, 4
        Y = rand(0:1, N, T)
        X = rand(1000:5000, N, T)
        B = rand(0:1, N)
        FV = randn(N, T)
        α = [2.0, -0.15, 1.0]
        
        like = likebus_ccp(α, Y, X, B, FV, N, T)
        
        @test isa(like, Float64)
        @test like > 0  # Likelihood should be positive
    end
end

# ECON 6343: Econometrics III 
# Student Name: Jacob Dison Wambede
# Date: 2024-28-09
# Problem Set 4: PS5-ddc
#Prof. Tyler Ransom
#Department of Economics
#University of Oklahoma

using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

# read in function to create state transitions for dynamic model
include("create_grids.jl")

function solve_ps5()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1: reshaping the data
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # load in the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))

    #---------------------------------------------------
    # reshape from wide to long (must do this twice be-
    # cause DataFrames.stack() requires doing it one 
    # variable at a time)
    #---------------------------------------------------
    # first reshape the decision variable
    dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfy_long, Not(:variable))

    # next reshape the odometer variable
    dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfx_long, Not(:variable))

    # join reshaped df's back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    sort!(df_long,[:bus_id,:time])

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2: estimate a static version of the model
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Estimate the static logit model
    static_model = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
    println("Static Model Results:")
    println(static_model)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3a: read in data for dynamic model
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url_dynamic = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df_dynamic = CSV.read(HTTP.get(url_dynamic).body, DataFrame)

    Y = Matrix(df_dynamic[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
    Odo = Matrix(df_dynamic[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    Xst = Matrix(df_dynamic[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    Zst = df_dynamic.Zst
    B = df_dynamic.Branded

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3b: generate state transition matrices
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    zval,zbin,xval,xbin,xtran = create_grids()

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3c-e: Dynamic model estimation
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @views @inbounds function dynamic_loglikelihood(θ)
        # Initialize parameters
        θ_0, θ_1, θ_2 = θ
        β = 0.9
        N, T = size(Y)

        # Initialize future value array
        FV = zeros(zbin*xbin, 2, T+1)

        # Backwards recursion
        for t in T:-1:1
            for b in 0:1
                for z in 1:zbin
                    for x in 1:xbin
                        row = x + (z-1)*xbin
                        
                        # Conditional value function for driving (v1t)
                        v1t = θ_0 + θ_1*xval[x] + θ_2*b + 
                              β * (xtran[row,:]' * FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                        
                        # Conditional value function for replacing (v0t)
                        v0t = β * (xtran[1+(z-1)*xbin,:]' * FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                        
                        # Update future value
                        FV[row, b+1, t] = β * log(exp(v0t) + exp(v1t))
                    end
                end
            end
        end

        # Compute log-likelihood
        loglik = 0.0
        for i in 1:N
            for t in 1:T
                row0 = 1 + (Zst[i]-1)*xbin
                row1 = Xst[i,t] + (Zst[i]-1)*xbin
                
                v1_v0 = θ_0 + θ_1*Odo[i,t] + θ_2*B[i] + 
                        β * ((xtran[row1,:].-xtran[row0,:])' * FV[row0:row0+xbin-1, B[i]+1, t+1])
                
                P1 = exp(v1_v0) / (1 + exp(v1_v0))
                loglik += Y[i,t] * log(P1) + (1-Y[i,t]) * log(1-P1)
            end
        end

        return -loglik  # Return negative log-likelihood for minimization
    end

    # Initial guess (using static model estimates)
    initial_θ = coef(static_model)

    # Optimize
    result = optimize(dynamic_loglikelihood, initial_θ, BFGS(), Optim.Options(show_trace=true))

    # Print results
    function allwrap()

    println("\nDynamic Model Results:")
    println("Estimated parameters: ", Optim.minimizer(result))
    println("Log-likelihood: ", -Optim.minimum(result))
    println("Converged: ", Optim.converged(result))
end

#call the main function
allwrap()

end
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4: Unit tests
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using Test

@testset "Bus Engine Replacement Model Tests" begin
    # Test data loading
    @test begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        !isempty(df)
    end

    # Test data reshaping
    @test begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        df = @transform(df, :bus_id = 1:size(df,1))
        dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20, :RouteUsage, :Branded)
        dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
        size(dfy_long, 1) == 20 * size(df, 1)
    end

    # Test static model estimation
    @test begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        df_long = bus_engine_replacement_model()  # Assuming this function returns the reshaped dataframe
        model = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
        !isnothing(model)
    end

    # Test dynamic model data loading
    @test begin
        url_dynamic = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
        df_dynamic = CSV.read(HTTP.get(url_dynamic).body, DataFrame)
        !isempty(df_dynamic)
    end
 

    # Test create_grids function
    @test begin
        zval, zbin, xval, xbin, xtran = create_grids()
        !isnothing(zval) && !isnothing(zbin) && !isnothing(xval) && !isnothing(xbin) && !isnothing(xtran)
    end

    # Test dynamic_loglikelihood function
    @test begin
        result = bus_engine_replacement_model()  # Assuming this function returns the optimization result
        !isnothing(result) && Optim.converged(result)
    end
end

# done with the script

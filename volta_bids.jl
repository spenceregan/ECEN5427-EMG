using CSV
using DataFrames
using CategoricalArrays
using StatsModels, GLM
using Random, Statistics, Distributions
using JuMP, HiGHS, Gurobi

rng = MersenneTwister()

wd = "/Users/spenceregan/"
datapath = joinpath(pwd(),"data")
myteam = "Volta"

function predict_demand(
    demand::DataFrame, 
    growth::AbstractFloat, 
    years::Integer
    )
    newdemand = demand .* (1+growth)^years
    return newdemand
end

function rindex(
    trends::DataFrame, 
    round::Integer
    )
    index = round - trends.round[1] + 1
    return index
end

function clean_demand_trends(trends::DataFrame)
    cr = maximum(trends.round)
    dem_cols = Dict(
        :offpeakmax => :offpeak,
        :shouldermax => :shoulder,
        :onpeakmax => :onpeak
        )
    rename!(trends, dem_cols)
    prev_demand = trends[trends.round .== (cr - 1), collect(values(dem_cols))]
    growth = trends[rindex(trends, (cr - 1)), :demandgrowth]
    trends[trends.round .== cr, collect(values(dem_cols))] = predict_demand(prev_demand, growth, 1)
    return trends
end

function planttypes_dict(planttypes::DataFrame)
    ptype_dict = Dict(
        "fuel" => Dict(Pair.(planttypes.type, planttypes.fuel)),
        "fuelenergy" => Dict(Pair.(planttypes.type, planttypes.energydensity)),
        "construction" => Dict(Pair.(planttypes.type, planttypes.constructionterm)),
        "CO2" => Dict(Pair.(planttypes.type, planttypes.emmissions))
    )
    return ptype_dict
end

function wind_avail(
    round::Integer, 
    trends::DataFrame,
    wind_beta::Beta
    )
    if round in trends.round
        avail = trends[rindex(trends, round), :windavail]
    else
        avail = quantile(wind_beta, rand())
    end
    return avail
end

function future_scenarios(trends::DataFrame, horizon::Integer)
    avg_growth = mean(trends.demandgrowth)
    curround = maximum(trends.round)
    currdemand = trends[trends.round .== curround, [:offpeak, :shoulder, :onpeak]]
    offpeak_predict(round) = (1 + avg_growth)^(round - curround) .* currdemand.offpeak[1]
    shoulder_predict(round) = (1 + avg_growth)^(round - curround) .* currdemand.shoulder[1]
    onpeak_predict(round) = (1 + avg_growth)^(round - curround) .* currdemand.onpeak[1]
    scenarios = DataFrame(round = curround:curround+horizon)
    transform!(scenarios, :round => ByRow(offpeak_predict)  => :offpeak)
    transform!(scenarios, :round => ByRow(shoulder_predict)  => :shoulder)
    transform!(scenarios, :round => ByRow(onpeak_predict)  => :onpeak)
    return scenarios
end

function efficiency_predict(plants::DataFrame, newplants::DataFrame, ptypes)
    effplants = select(plants, :commissioned, :type, :efficiency)
    effnewplants = select(newplants, :round, :type, :efficiency)

    comm_round(invest, type) = [ptypes["construction"][t] for t in type] .+ invest

    transform!(effnewplants, [:round, :type] => comm_round => :commissioned)
    select!(effnewplants, :commissioned, :type, :efficiency)
    append!(effplants, effnewplants)
    dropmissing!(effplants)

    effformula = @formula(efficiency ~ type + commissioned&type)
    effmodel = lm(effformula, effplants)
    plants.eff_predict = predict(effmodel, plants)
    eff_missing(A,B) = [ismissing(a) ? b : a for (a,b) in zip(A,B)]
    transform!(plants, [:efficiency, :eff_predict] => eff_missing => :efficiency)
    select!(plants, Not(:eff_predict))
    return plants
end

function mybids(
    round::Integer, 
    plants::DataFrame, 
    planthistory::DataFrame, 
    trends::DataFrame,
    ptypes::Dict
    )
    commissioned(id) = plants[plants.id .== id, :commissioned][1]
    decommissioned(id) = plants[plants.id .== id, :decommissioned][1]
    eff(id) = plants[plants.id .== id, :efficiency][1]
    type(id) = plants[plants.id .== id, :type][1]
    fuel_price(type) = trends[trends.round .== round, ptypes["fuel"][type]][1]

    wind_availability = trends[trends.round .== round, :windavail][1]

    function available_capacity(id, type, operational)
        qty = plants[plants.id .== id, :capacity][1]
        qty *= (type == "wind" ? wind_availability : 1)
        qty *= operational
        return qty
    end

    function marginal_cost(type, efficiency)
        fuel_price(type) / ptypes["fuelenergy"][type] / efficiency * 3600
    end

    myplants = subset(planthistory, :round => r -> r .== round)

    subset!(myplants, :id => ByRow(p -> commissioned(p) .<= round))
    subset!(myplants, :id => ByRow(p -> decommissioned(p) .> round))
    transform!(myplants, :id => ByRow(type) => :id_type)
    transform!(myplants, :id => ByRow(eff) => :efficiency)
    transform!(myplants, [:id_type, :efficiency] => ByRow(marginal_cost) => :price)
    transform!(myplants, [:id, :id_type, :operational] => ByRow(available_capacity) => :qty)

    bids = select!(combine(groupby(myplants, :price), :qty => sum), :qty_sum, :price)

    CSV.write(joinpath(pwd(), "bids", string("R", round, "_ebid_Volta.csv")), bids)
    return bids
end

function reliability_model(
    plants::DataFrame, 
    planthistory::DataFrame, 
    newplants::DataFrame
    )
    rnewplants = select(newplants, :type, :reliability)
    rplants = select(planthistory, :id, :round, :reliability)

    commissioned(id) = plants[plants.id .== id, :commissioned][1]
    ptype(id) = plants[plants.id .== id, :type][1]
    age(round, commissioned) = maximum([0, round - commissioned])

    transform!(rplants, :id => ByRow(ptype) => :type)
    transform!(rplants, :id => ByRow(commissioned) => :commissioned)
    transform!(rplants, [:round, :commissioned] => ByRow(age) => :age)

    select!(rplants, :age, :type, :reliability)

    rnewplants.age .= 0
    append!(rplants, rnewplants)

    reliformula = @formula(reliability ~ type + age&type)
    relimodel = lm(reliformula, rplants)

    return relimodel
end

function reliability_predict(plants, round, relimodel)
    age(commissioned) = maximum([0, round - commissioned])
    transform!(plants, :commissioned => ByRow(age) => :age)
    plants.reliability = predict(relimodel, plants)
    select!(plants, Not(:age))
    return plants
end

function fleet_bids(
    plants::DataFrame, 
    round::Integer, 
    trends::DataFrame, 
    ptypes::Dict,
    )
    fuelprice(type) = trends[rindex(trends, round), ptypes["fuel"][type]]

    activeplants = subset!(plants, :commissioned => c -> c .<= round)
    subset!(activeplants, :decommissioned => d -> d .> round)
   
    function mc(type, efficiency)
        fuelprice(type) / ptypes["fuelenergy"][type] / efficiency * 3600
    end

    transform!(activeplants, [:type, :efficiency] => ByRow(mc) => :price)
    select!(activeplants, :id, :type, :reliability, :capacity, :price)

    return activeplants
end

function solve_ed(bids::DataFrame, maxdem)
    capacities = bids[:, :qty]
    costs = bids[:, :price]
    Nplants = nrow(bids)
    ed = Model(Gurobi.Optimizer)
    @variable(ed, zeros(Nplants)[i] <= P[i = 1:Nplants] <= capacities[i])
    @variable(ed, 0 <= demand <= maxdem)
    cons_value(d) = 0.5 * 0.1 * (maxdem^2 - (maxdem-d)^2)
    @objective(ed, Max, cons_value(demand) - costs'*P)
    @constraint(ed, pbal, ones(Nplants)' * P == demand)
    display(ed)
    optimize!(ed)
    # dispatch = value.(P)
    # cleared_demand = value.(demand)
    clearing_price = dual(pbal)
    return clearing_price
end

function clearing_prices(
    fleet::DataFrame, 
    scenarios::DataFrame, 
    planthistory::DataFrame,
    trends::DataFrame,
    wind_beta::Beta
    )
    n = 1000
    Nrounds = nrow(scenarios)
    prices = DataFrame(round = zeros(n * Nrounds), offpeak = zeros(n * Nrounds), shoulder = zeros(n * Nrounds), onpeak = zeros(n * Nrounds))
    subset!(fleet, :type => ByRow(t -> t in ["wind", "nuclear"]))
    function expected_op(reliability)
        op = (reliability .> rand(rng, length(reliability)))
        return op
    end
    wind = Float64[]

    for r in 1:Nrounds
        round = scenarios.round[r]

        function overwrite_operational(id, expected_op)
            phistory = subset(planthistory, :id => i -> i .== id)
            subset!(phistory, :round => r -> r .== round)
            op = (nrow(phistory)>0) ? phistory[1, :operational] : expected_op
            return op
        end

        for i in 1:n
            fplants = transform(fleet, :reliability => expected_op => :operating)
            wind_availability = wind_avail(round, trends, wind_beta)
            push!(wind, wind_availability)
            qty(type, cap) = cap * (type == "wind" ? wind_availability : 1)
            transform!(fplants, [:type, :capacity] => ByRow(qty) => :qty)
            transform!(fplants, [:id, :operating] => ByRow(overwrite_operational) => :operating)
            subset!(fplants, :operating => c -> c .== true)
            for per in [:offpeak, :shoulder, :onpeak]
                period_demand = scenarios[r, per]
                price = solve_ed(fplants, period_demand)
                prices[i + (r-1)*n, per] = price
                prices[i + (r-1)*n, :round] = round
            end
        end
    end
    periods = [:offpeak, :shoulder, :onpeak]
    Eprices = groupby(prices, :round)
    prices_summary = combine(Eprices, periods .=> mean, periods .=> std)
    display(wind)
    return prices_summary
end

function main()
    plants = DataFrame(CSV.File(joinpath(datapath, "plant_summary.csv")))
    dropmissing!(plants, :id)
    planthistory = DataFrame(CSV.File(joinpath(datapath, "plant_history.csv")))
    newplants = DataFrame(CSV.File(joinpath(datapath, "new_plants.csv")))
    trends = DataFrame(CSV.File(joinpath(datapath, "trends.csv")))

    trends = clean_demand_trends(trends)
    wind_beta = fit(Beta, trends.windavail)

    planttypes = DataFrame(CSV.File(joinpath(datapath, "plant_types.csv")))
    ptypes = planttypes_dict(planttypes)

    currentrnd = maximum(trends.round)

    plants = efficiency_predict(plants, newplants, ptypes)

    reliamodel = reliability_model(plants, planthistory, newplants)
    currplants = reliability_predict(plants, currentrnd, reliamodel)

    mybids(currentrnd, currplants, planthistory, trends, ptypes)

    fleetbids = fleet_bids(currplants, currentrnd, trends, ptypes)

    scenario = future_scenarios(trends, 5)
    prices = clearing_prices(fleetbids, scenario, planthistory, trends, wind_beta)
    return prices
end



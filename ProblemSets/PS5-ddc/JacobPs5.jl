# Consumption Baskets and PPP Exchange Rate Calculations

"""
Calculate Argentina's GDP in dollars using different exchange rate methods
"""
function calculate_gdp()
    # Given data
    # Quantities and prices
    us_grain_q = arg_grain_q = 10
    us_energy_q = arg_energy_q = 10
    
    us_grain_p = 1.0  # dollars
    us_energy_p = 2.0  # dollars
    arg_grain_p = 1.0  # pesos
    arg_energy_p = 2.0  # pesos
    
    # Basket compositions
    us_basket_grain = 1.0
    us_basket_energy = 1.0
    arg_basket_grain = 1.5
    arg_basket_energy = 0.5
    
    # (a) GDP at market exchange rates
    function market_exchange_rate()
        println("\n(a) GDP at market exchange rates:")
        
        # Since law of one price holds, exchange rate = 1
        E = 1.0
        println("Market exchange rate = $(E) peso/dollar")
        
        # Calculate Argentina's GDP in pesos
        arg_gdp_pesos = arg_grain_q * arg_grain_p + arg_energy_q * arg_energy_p
        println("Argentina GDP in pesos = $(arg_gdp_pesos)")
        
        # Convert to dollars
        arg_gdp_dollars = arg_gdp_pesos * (1/E)
        println("Argentina GDP in dollars = \$$(arg_gdp_dollars)")
        
        return arg_gdp_dollars
    end
    
    # (b) GDP at PPP rates using own baskets
    function ppp_own_baskets()
        println("\n(b) GDP at PPP rates using own baskets:")
        
        # Calculate cost of baskets
        us_basket_dollars = us_basket_grain * us_grain_p + us_basket_energy * us_energy_p
        println("Cost of US basket in dollars = \$$(us_basket_dollars)")
        
        arg_basket_pesos = arg_basket_grain * arg_grain_p + arg_basket_energy * arg_energy_p
        println("Cost of Argentina basket in pesos = $(arg_basket_pesos)")
        
        # Calculate PPP exchange rate
        E_ppp = arg_basket_pesos / us_basket_dollars
        println("PPP exchange rate = $(E_ppp) pesos/dollar")
        
        # Calculate Argentina's GDP
        arg_gdp_pesos = arg_grain_q * arg_grain_p + arg_energy_q * arg_energy_p
        arg_gdp_dollars = arg_gdp_pesos * (1/E_ppp)
        println("Argentina GDP in dollars = \$$(arg_gdp_dollars)")
        
        return arg_gdp_dollars
    end
    
    # (c) GDP at PPP rates using US basket
    function ppp_us_basket()
        println("\n(c) GDP at PPP rates using US basket:")
        
        # Calculate cost of US basket in both currencies
        us_basket_dollars = us_basket_grain * us_grain_p + us_basket_energy * us_energy_p
        us_basket_pesos = us_basket_grain * arg_grain_p + us_basket_energy * arg_energy_p
        
        println("Cost of US basket in dollars = \$$(us_basket_dollars)")
        println("Cost of US basket in pesos = $(us_basket_pesos)")
        
        # Calculate PPP exchange rate
        E_ppp = us_basket_pesos / us_basket_dollars
        println("PPP exchange rate = $(E_ppp) pesos/dollar")
        
        # Calculate Argentina's GDP
        arg_gdp_pesos = arg_grain_q * arg_grain_p + arg_energy_q * arg_energy_p
        arg_gdp_dollars = arg_gdp_pesos * (1/E_ppp)
        println("Argentina GDP in dollars = \$$(arg_gdp_dollars)")
        
        return arg_gdp_dollars
    end
    
    # (d) GDP at PPP rates using Argentine basket
    function ppp_arg_basket()
        println("\n(d) GDP at PPP rates using Argentine basket:")
        
        # Calculate cost of Argentine basket in both currencies
        arg_basket_dollars = arg_basket_grain * us_grain_p + arg_basket_energy * us_energy_p
        arg_basket_pesos = arg_basket_grain * arg_grain_p + arg_basket_energy * arg_energy_p
        
        println("Cost of Argentine basket in dollars = \$$(arg_basket_dollars)")
        println("Cost of Argentine basket in pesos = $(arg_basket_pesos)")
        
        # Calculate PPP exchange rate
        E_ppp = arg_basket_pesos / arg_basket_dollars
        println("PPP exchange rate = $(E_ppp) pesos/dollar")
        
        # Calculate Argentina's GDP
        arg_gdp_pesos = arg_grain_q * arg_grain_p + arg_energy_q * arg_energy_p
        arg_gdp_dollars = arg_gdp_pesos * (1/E_ppp)
        println("Argentina GDP in dollars = \$$(arg_gdp_dollars)")
        
        return arg_gdp_dollars
    end
    
    # Calculate all results
    println("Complete Solution for Argentina's GDP Calculations:")
    a = market_exchange_rate()
    b = ppp_own_baskets()
    c = ppp_us_basket()
    d = ppp_arg_basket()
    
    # Summary
    println("\nSummary of Results:")
    println("Method (a) Market exchange rates: \$$(a)")
    println("Method (b) PPP using own baskets: \$$(b)")
    println("Method (c) PPP using US basket: \$$(c)")
    println("Method (d) PPP using Argentine basket: \$$(d)")
end

# Run the calculations
calculate_gdp()
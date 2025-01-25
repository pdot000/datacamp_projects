Analyze the experimental setup and identify significant patterns and insights.

- Determine the experimental design type used by the environmental research team, storing either `"factorial"` or `"randomized_block"` in the `design` string object accordingly.
- Create a visualization by plotting the CO2 emissions against each geographical region colored by fuel source. Which combination of region and fuel source has the highest median CO2 emissions value? Store your responses in the `highest_co2_region` and `highest_co2_source` string objects.
- Is there a significant difference in CO2 emissions based on fuel source, grouped by region? Assume a significance level 0.05 and store your results as the `test_results` `pandas` Series.
- If a significant difference is found, determine if there are any pairwise significant differences using post-hoc analysis, storing the comparisons data in the `diff_results` tuple.
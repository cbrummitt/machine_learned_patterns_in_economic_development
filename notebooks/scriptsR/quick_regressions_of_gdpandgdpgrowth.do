

import delimited "C:\Users\agomez\Dropbox\Identify laws and dynamical systems in socioeconomic data\machine_learned_patterns_in_economic_development\notebooks\data_output\final_dt.csv", clear 

bysort country_code (year): gen delta_gdp_1 = log10_gdppercap[_n + 1] - log10_gdppercap[_n] if year[_n+1] - year[_n] == 1 
bysort country_code (year): gen delta_gdp_10 = (1 + log10_gdppercap[_n + 10] - log10_gdppercap[_n])^(1/10) - 1 if year[_n+10] - year[_n] ==10

egen group_year  = group(year)

eststo clear
qui eststo: reg log10_gdppercap log10_exp_percapita                            , vce(cluster country_code)
qui eststo: reg log10_gdppercap                     pc0_year                   , vce(cluster country_code)
qui eststo: reg log10_gdppercap                              sqrt_diversity    , vce(cluster country_code)
qui eststo: reg log10_gdppercap                                             eci, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita          sqrt_diversity    , vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita                         eci, vce(cluster country_code)
qui eststo: reg log10_gdppercap                     pc0_year sqrt_diversity    , vce(cluster country_code)
qui eststo: reg log10_gdppercap                     pc0_year                eci, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year                   , vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year sqrt_diversity    , vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year                eci, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year sqrt_diversity eci, vce(cluster country_code)
esttab , r2 ar2 noconstant compress nogaps //wide


eststo clear
qui eststo: reg log10_gdppercap log10_exp_percapita                             i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap                     pc0_year                    i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap                              sqrt_diversity     i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap                                             eci i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita          sqrt_diversity     i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita                         eci i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap                     pc0_year sqrt_diversity     i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap                     pc0_year                eci i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year                    i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year sqrt_diversity     i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year                eci i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year sqrt_diversity eci i.year, vce(cluster country_code)
esttab , r2 ar2 noconstant compress nogaps //wide

eststo clear
qui eststo: reg log10_gdppercap log10_exp_percapita                             if year==1995, vce(cluster country_code)
qui eststo: reg log10_gdppercap                     pc0_year                    if year==1995, vce(cluster country_code)
qui eststo: reg log10_gdppercap                              sqrt_diversity     if year==1995, vce(cluster country_code)
qui eststo: reg log10_gdppercap                                             eci if year==1995, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita          sqrt_diversity     if year==1995, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita                         eci if year==1995, vce(cluster country_code)
qui eststo: reg log10_gdppercap                     pc0_year sqrt_diversity     if year==1995, vce(cluster country_code)
qui eststo: reg log10_gdppercap                     pc0_year                eci if year==1995, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year                    if year==1995, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year sqrt_diversity     if year==1995, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year                eci if year==1995, vce(cluster country_code)
qui eststo: reg log10_gdppercap log10_exp_percapita pc0_year sqrt_diversity eci if year==1995, vce(cluster country_code)
esttab , r2 ar2 noconstant compress nogaps //wide

* pc0 pulls the signs of diversity and compexity to the opposite sign, suggesting pc0 is both a measure of diversity and complexity


eststo clear
qui eststo: reg delta_gdp_10 log10_gdppercap                                                , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita                            , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                     pc0_year                   , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                              sqrt_diversity    , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                                             eci, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita          sqrt_diversity    , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita                         eci, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                     pc0_year sqrt_diversity    , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                     pc0_year                eci, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita pc0_year                   , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita pc0_year sqrt_diversity    , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita pc0_year                eci, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita pc0_year sqrt_diversity eci, vce(cluster country_code)
esttab , r2 ar2 noconstant compress nogaps //wide

reg delta_gdp_10 log10_gdppercap log10_exp_percapita pc0_year sqrt_diversity eci i.year if year==1965 | year==1975 | year==1985 | year==1995 | year==2005, vce(cluster country_code)

* with year fixed effects
eststo clear
qui eststo: reg delta_gdp_10 log10_gdppercap                                                 i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita                             i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                     pc0_year                    i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                              sqrt_diversity     i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                                             eci i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita          sqrt_diversity     i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita                         eci i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                     pc0_year sqrt_diversity     i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                     pc0_year                eci i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita pc0_year                    i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita pc0_year sqrt_diversity     i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita pc0_year                eci i.year, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap log10_exp_percapita pc0_year sqrt_diversity eci i.year, vce(cluster country_code)
esttab , r2 ar2 noconstant compress nogaps //wide












eststo clear
qui eststo: reg log10_gdppercap pc0_year                                                                 , vce(cluster country_code)
qui eststo: reg log10_gdppercap          pc1_year                                                        , vce(cluster country_code)
qui eststo: reg log10_gdppercap                   sqrt_diversity                                         , vce(cluster country_code)
qui eststo: reg log10_gdppercap                                  log10_exp_percapita                     , vce(cluster country_code)
qui eststo: reg log10_gdppercap                                                      log10_population    , vce(cluster country_code)
qui eststo: reg log10_gdppercap                                                                       eci, vce(cluster country_code)
qui eststo: reg log10_gdppercap          pc1_year sqrt_diversity                     log10_population eci, vce(cluster country_code)
qui eststo: reg log10_gdppercap pc0_year pc1_year sqrt_diversity                     log10_population eci, vce(cluster country_code)
qui eststo: reg log10_gdppercap          pc1_year sqrt_diversity log10_exp_percapita log10_population eci, vce(cluster country_code)
qui eststo: reg log10_gdppercap pc0_year pc1_year sqrt_diversity log10_exp_percapita log10_population eci, vce(cluster country_code)
qui eststo: reg log10_gdppercap pc0_year          sqrt_diversity log10_exp_percapita                  eci, vce(cluster country_code)
esttab , r2 ar2 noconstant compress nogaps //wide

cor log10_gdppercap pc0_year pc1_year sqrt_diversity log10_exp_percapita log10_population eci if year==2010
reg log10_gdppercap pc0_year pc1_year sqrt_diversity log10_exp_percapita log10_population eci, vce(cluster country_code)




eststo clear
qui eststo: reg log10_gdppercap                                                                           i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap pc0_year                                                                  i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap          pc1_year                                                         i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap                   sqrt_diversity                                          i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap                                  log10_exp_percapita                      i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap                                                      log10_population     i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap                                                                 eci       i.year, vce(cluster country_code)
qui eststo: reg log10_gdppercap pc0_year pc1_year sqrt_diversity log10_exp_percapita log10_population eci i.year, vce(cluster country_code)
esttab , r2 ar2 noconstant compress nogaps //wide

eststo clear
qui eststo: reg change_log10_gdppercap log10_gdppercap pc0_year                                                                 , vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap          pc1_year                                                        , vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap                   sqrt_diversity                                         , vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap                                  log10_exp_percapita                     , vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap                                                      log10_population    , vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap                                                                 eci      , vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap pc0_year pc1_year sqrt_diversity log10_exp_percapita log10_population eci, vce(cluster country_code)
esttab , r2 ar2 noconstant compress nogaps //wide

eststo clear
qui eststo: reg change_log10_gdppercap log10_gdppercap                                                                           i.year, vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap pc0_year                                                                  i.year, vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap          pc1_year                                                         i.year, vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap                   sqrt_diversity                                          i.year, vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap                                  log10_exp_percapita                      i.year, vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap                                                      log10_population     i.year, vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap                                                                 eci       i.year, vce(cluster country_code)
qui eststo: reg change_log10_gdppercap log10_gdppercap pc0_year pc1_year sqrt_diversity log10_exp_percapita log10_population eci i.year, vce(cluster country_code)
esttab , r2 ar2 noconstant compress nogaps //wide





eststo clear
qui eststo: reg delta_gdp_10 log10_gdppercap pc0_year                                                                 , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap          pc1_year                                                        , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                   sqrt_diversity                                         , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                                  log10_exp_percapita                     , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                                                      log10_population    , vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap                                                                       eci, vce(cluster country_code)
qui eststo: reg delta_gdp_10 log10_gdppercap pc0_year pc1_year sqrt_diversity log10_exp_percapita log10_population eci, vce(cluster country_code)
esttab , r2 ar2 noconstant compress nogaps //wide










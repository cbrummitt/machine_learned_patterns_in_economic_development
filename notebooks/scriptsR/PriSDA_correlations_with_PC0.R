# PriSDA_correlations_with_PC0.R


rm(list = ls(all = TRUE))  # resets R to fresh
gc()
pryr::mem_used()

# =======================================================
# DECLARATION OF LIBRARIES
# -------------------------------------------------------

packagelist <- c("viridis", "ggbeeswarm", "ggplot2", "ggridges", "sfsmisc", "scales",
                 "broom", "dplyr", "tidyr", "data.table",
                 "lfe", "stargazer")
# trace(utils:::unpackPkgZip,edit = T)
if (!require("pacman")) install.packages("pacman")
pacman::p_load(char = packagelist)


.pardefault <- par()


# -------------------------------------------------------


# =======================================================
# DECLARATION OF PATHS
# -------------------------------------------------------
path_to_workingfolder <- dirname(rstudioapi::getActiveDocumentContext()$path)

path_to_figures <- paste0(path_to_workingfolder,"/../regression_figures/")

path_to_data <- paste0(path_to_workingfolder, "/../../data/")
path_to_raw_data <- paste0(path_to_data, "raw/")
path_to_processed_data <- paste0(path_to_data, "processed/")
path_to_external_data <- paste0(path_to_workingfolder,"/../data_external/")
path_to_output_data <- paste0(path_to_workingfolder,"/../data_output/")


# -------------------------------------------------------


# =======================================================
# LOADING DATA
# -------------------------------------------------------

## Principal components scores and growth
pc_scores_growth_dt <- fread(paste0(path_to_processed_data, 
                                    "Rpop__data_target__pca_2__target_is_difference_True.csv"))

# making "last_year" to be current year, with "change" into the future
pc_scores_growth_dt[, last_year := lag(year)]
pc_scores_growth_dt[, year := NULL]

colnames(pc_scores_growth_dt) <- sub("last_", "", colnames(pc_scores_growth_dt))

setkey(pc_scores_growth_dt, country_code, year)


## Worlwide Governance Indicators
wgi_dt_list <- list()
for(sheet_i in c(2:7)){
  wgi_dt <- rio::import(paste0(path_to_external_data, 
                               "wgidataset_v2.xlsx"), sheet=sheet_i)
  names_of_columns <- colnames(wgi_dt)[3:length(colnames(wgi_dt))]
  wgi_long_dt <- wgi_dt %>%
    gather(column_name, column_value, c(names_of_columns)) %>% 
    extract(column_name, into=c("var_name", "year"), regex="([[:alpha:]]+)([[:digit:]]+)") %>% 
    filter(grepl("([[:alpha:]]+)(Estimate)([[:digit:]]+)?", var_name) & column_value!="#N/A") %>%
    mutate(year = as.integer(year),
           column_value = as.numeric(column_value)) %>%
    arrange(WBCode, year) %>%
    as.data.table()

  name_of_data <- wgi_long_dt$var_name[1]
  print(name_of_data)
  
  wgi_dt_list[[name_of_data]] <- wgi_long_dt
}

# Appending all together
wgi_final_dt <- rbindlist(wgi_dt_list, 
                          use.names = TRUE,
                          fill=TRUE)

# Spreading to have only country year rows
wgi_final_dt <- wgi_final_dt %>%
  spread(var_name, column_value) %>%
  arrange(WBCode, year) %>%
  as.data.table()

colnames(wgi_final_dt)[colnames(wgi_final_dt)=="WBCode"] <- "country_code"

setkey(wgi_final_dt, country_code, year)


# clean memory and garbage collect
rm(wgi_dt, wgi_long_dt, wgi_dt_list)
gc() 
pryr::mem_used()


## Years of schooling
yrsch_dt <- as.data.table(rio::import(paste0(path_to_external_data, 
                                             "BL2013_MF1599_v2.2.csv")))
colnames(yrsch_dt)[colnames(yrsch_dt)=="WBcode"] <- "country_code"
cols2keep <- c("country_code", "year", "yr_sch")
yrsch_dt <- yrsch_dt[, .SD, .SDcols = c(cols2keep)]

setkey(yrsch_dt, country_code, year)


## Cognitive abilities
cognitive_dt <- as.data.table(rio::import(paste0(path_to_external_data, 
                                                 "hanushek+woessmann.cognitive.xls"), sheet=2))
colnames(cognitive_dt)[colnames(cognitive_dt)=="Code"] <- "country_code"
cols2keep <- c("country_code", "cognitive")
cognitive_dt <- cognitive_dt[, .SD, .SDcols = c(cols2keep)]

setkey(cognitive_dt, country_code)


## Economic Complexity Index and 4-digit product diversity
ecidiv_dt <- as.data.table(rio::import(paste0(path_to_raw_data, "exports/", 
                          "S2_final_cpy_all.dta")))
colnames(ecidiv_dt)[colnames(ecidiv_dt)=="commoditycode"] <- "product_code"
colnames(ecidiv_dt)[colnames(ecidiv_dt)=="exporter"] <- "country_code"

ecidiv_dt <- ecidiv_dt %>%
  group_by(country_code, year) %>%
  dplyr::summarise(eci = first(eci),
                   diversity = sum(mcp, na.rm=TRUE),
                   population = first(population),
                   total_exports_mm = sum(export_value/10^6),
                   exports_percapita = if_else(population>0 & total_exports_mm>0, 
                                               (total_exports_mm/population)*10^6,
                                               NULL)) %>%
  arrange(country_code, year) %>%
  as.data.table()

setkey(ecidiv_dt, country_code, year)

# clean memory and garbage collect
gc() 
pryr::mem_used()


# -------------------------------------------------------


# =======================================================
# Merging all datasets
# -------------------------------------------------------
cols2match <- c("country_code", "year")

final_dt <- cognitive_dt[yrsch_dt[wgi_final_dt[pc_scores_growth_dt[ecidiv_dt, on=c(cols2match)], 
                                                       on=c(cols2match)], 
                                          on=c(cols2match)], 
                          on="country_code"]
setorder(final_dt, country_code, year)

rm(cognitive_dt, ecidiv_dt, pc_scores_growth_dt, wgi_final_dt, yrsch_dt)
gc()
pryr::mem_used()

final_dt %>% 
  group_by(year) %>%
  dplyr::summarise(numcountries = n_distinct(country_code)) %>%
  as.data.table()

# -------------------------------------------------------


# =======================================================
# NEW COLUMNS
# -------------------------------------------------------
final_dt[, c("log10_population",
             "sqrt_diversity",
             "log10_total_exports_mm",
             "log10_exp_percapita") := list(
               log10(population[population>0]),
               sqrt(diversity),
               log10(total_exports_mm[total_exports_mm>0]),
               log10(exports_percapita[exports_percapita>0])
             )]


cnames <- colnames(final_dt)
cnames[cnames=="log10_gdp_per_capita_constant2010USD_year"] <- "log10_gdppercap"
cnames[cnames=="change_in_log10_gdp_per_capita_constant2010USD"] <- "change_log10_gdppercap"
colnames(final_dt) <- cnames

rio::export(final_dt,
            paste0(path_to_output_data, 
                   "final_dt.csv"))

# -------------------------------------------------------


# =======================================================
# STANDARDIZING THE VARIABLES
# -------------------------------------------------------
allcols <- colnames(final_dt)
cols2standardize <- c()
for(col_i in allcols){
  cat(substr(col_i, 1,min(10, nchar(col_i))))
  callstr <- paste0("final_dt$", col_i)
  
  vec <- eval(parse(text = callstr))
  
  cat("\t\t ")
  
  # is the column standardizable? i.e., is it numeric?
  standardizable <- (is.numeric(vec) & col_i!="year")
   
  cat(standardizable)
  cat("\n")
  
  if(standardizable){
    cols2standardize <- c(cols2standardize, col_i)
    #final_dt[, c(paste0("s_", col_i)) := list()]
  }
  
}

# Dataset where variables have been standardized
s_final_dt <- final_dt %>% 
  group_by(year) %>% 
  mutate_at(funs(scale(.) %>% as.vector), 
                             .vars=c(cols2standardize)) %>%
  arrange(country_code, year) %>%
  as.data.table()

s_final_dt[, c("country_code", "year") := list(as.factor(country_code), 
                                               factor(year, levels = as.character(seq(1962, 2016,1))))]

cnames <- colnames(s_final_dt)
cnames[cnames=="log10_gdp_per_capita_constant2010USD_year"] <- "log10_gdppercap"
cnames[cnames=="change_in_log10_gdp_per_capita_constant2010USD"] <- "change_log10_gdppercap"
colnames(s_final_dt) <- cnames

rio::export(s_final_dt,
            paste0(path_to_output_data, 
                   "s_final_dt.csv"))

# -------------------------------------------------------


# =======================================================
# REGRESSING PC0 ON OTHERS
# If we include all years, we need to cluster the errors
# since residuals will be correlated within countries
# across years, and t-values will be inflated.

# load necessary packages for clustering errors
library(multiwayvcov)
library(lmtest)

# -------------------------------------------------------
allvars <- c("log10_exp_percapita", "sqrt_diversity", "eci",
             "ControlOfCorruptEstimate", "GovEffEstimate",
             "PolStabEstimate", "RegQualityEstimate",
             "RuleOfLawEstimate", "VoiceAndAccbtyEstimate",
             "yr_sch", "cognitive")
uni_tidy_models <- tibble()
for(onevar in allvars){
  ### Model ith
  vars <- c("pc0_year", onevar, "year")
  ols_formula <- as.formula(paste(vars[1], paste(vars[2:length(vars)], collapse = " + "), sep = " ~ "))
  ols_i <- lm(formula = ols_formula, 
              data=na.omit(s_final_dt[, .SD, .SDcols = c("country_code", vars)]))
  
  # Cluster by country
  vcov_ols_i <- cluster.vcov(ols_i, na.omit(s_final_dt[, .SD, .SDcols = c("country_code", vars)])$country_code)
  cl_ols_i <- coeftest(ols_i, vcov_ols_i)

  tidy_ols_i <- tidy(cl_ols_i) %>% mutate(model = paste0("Model ", onevar), regtype = "Univariate")
  
  uni_tidy_models <- rbind(uni_tidy_models, tidy_ols_i)
  
}

### Model small MULTIVARIATE
vars <- c("pc0_year", "log10_exp_percapita", "sqrt_diversity", "eci", "year")
ols_formula <- as.formula(paste(vars[1], paste(vars[2:length(vars)], collapse = " + "), sep = " ~ "))
ols_small_mult <- lm(formula = ols_formula, 
               data=na.omit(s_final_dt[, .SD, .SDcols = c("country_code", vars)]))

# normal ols
summary(ols_small_mult)
# Cluster by country
vcov_ols_small_mult <- cluster.vcov(ols_small_mult, na.omit(s_final_dt[, .SD, .SDcols = c("country_code", vars)])$country_code)
cl_ols_small_mult <- coeftest(ols_small_mult, vcov_ols_small_mult)
(tidy_ols_small_mult <- tidy(cl_ols_small_mult) %>% mutate(model = "Model Multivariate", regtype = "Multivariate"))



### Model MULTIVARIATE
vars <- c("pc0_year", allvars, "year")
ols_formula <- as.formula(paste(vars[1], paste(vars[2:length(vars)], collapse = " + "), sep = " ~ "))
ols_mult <- lm(formula = ols_formula, 
            data=na.omit(s_final_dt[, .SD, .SDcols = c("country_code", vars)]))

# normal ols
summary(ols_mult)
# Cluster by country
vcov_ols_mult <- cluster.vcov(ols_mult, na.omit(s_final_dt[, .SD, .SDcols = c("country_code", vars)])$country_code)
cl_ols_mult <- coeftest(ols_mult, vcov_ols_mult)
(tidy_ols_mult <- tidy(cl_ols_mult) %>% mutate(model = "Model Multivariate", regtype = "Multivariate"))






# -------------------------------------------------------


# =======================================================
# VISUALIZING
library(dotwhisker)
# -------------------------------------------------------


predictors_names_old <- c(allvars, 
                          paste0("year", seq(1962, 2016, 1)))
predictors_names_new <- c("log10(Exports per capita)",
                          expression("Diversity^0.5"),
                          "Economic Complexity Index",
                          "Control of Corruption",
                          "Government Effectiveness",
                          "Political Stability and Absence of Violence",
                          "Regulatory Quality",
                          "Rule of Law",
                          "Voice and Accountability",
                          "Years of Schooling",
                          "Cognitive Skills",
                          paste0("Fixed Effect ", seq(1962, 2016, 1)))
names(predictors_names_new) <- predictors_names_old

uni_tidy_models <- uni_tidy_models %>% relabel_predictors(predictors_names_new)
tidy_ols_small_mult <- tidy_ols_small_mult %>% relabel_predictors(predictors_names_new)
tidy_ols_mult <- tidy_ols_mult %>% relabel_predictors(predictors_names_new)

# all_tidy_models <- rbind(uni_tidy_models, tidy_ols_mult) %>% mutate(regtype = as.factor(regtype))

dwplot(uni_tidy_models)
dwplot(tidy_ols_small_mult)
dwplot(tidy_ols_mult)


ggplt_univariate <- dwplot(uni_tidy_models, 
       order_vars = predictors_names_new[1:11],
       vline = geom_vline(xintercept = 0, colour = "grey60", linetype = 2)) + # plot line at zero _behind_ coefs
  theme_light() +
  xlab("Standardized Coefficient Estimates") + ylab("") +
  geom_vline(xintercept = 0, colour = "grey60", linetype = 2) +
  ggtitle(expression(paste("Explaining ", phi[0], " - Univariate Regressions"))) +
  theme(plot.title = element_text(face="bold"),
        legend.position ="none",
        legend.justification = c(0, 1),
        legend.background = element_rect(colour="grey80"),
        legend.title = element_blank())

ggplt_smallmultivariate <- dwplot(tidy_ols_small_mult, 
       order_vars = predictors_names_new[1:3],
       vline = geom_vline(xintercept = 0, colour = "grey60", linetype = 2)) + # plot line at zero _behind_ coefs
  theme_light() +
  xlab("Standardized Coefficient Estimates") + ylab("") +
  geom_vline(xintercept = 0, colour = "grey60", linetype = 2) +
  ggtitle(expression(paste("Explaining ", phi[0], " - Multivariate Regressions"))) +
  theme(plot.title = element_text(face="bold"),
        legend.position = "none",
        legend.justification = c(0, 1),
        legend.background = element_rect(colour="grey80"),
        legend.title = element_blank())

ggplt_allmultivariate <- dwplot(tidy_ols_mult, 
       order_vars = predictors_names_new[1:11],
       vline = geom_vline(xintercept = 0, colour = "grey60", linetype = 2)) + # plot line at zero _behind_ coefs
  theme_light() +
  xlab("Standardized Coefficient Estimates") + ylab("") +
  geom_vline(xintercept = 0, colour = "grey60", linetype = 2) +
  ggtitle(expression(paste("Explaining ", phi[0], " - Multivariate Regressions"))) +
  theme(plot.title = element_text(face="bold"),
        legend.position = "none",
        legend.justification = c(0, 1),
        legend.background = element_rect(colour="grey80"),
        legend.title = element_blank())




# ----------------
# For plotting
namefile <- paste("PC0_regressions_coefficients_",
                  "univariate",
                  ".pdf", sep="")
cairo_pdf(paste(path_to_figures,namefile,sep=""), 
          height=3.6,
          width=7,
          bg="transparent")
print(ggplt_univariate)
dev.off()

# ----------------
# For plotting
namefile <- paste("PC0_regressions_coefficients_",
                  "smallmultivariate",
                  ".pdf", sep="")
cairo_pdf(paste(path_to_figures,namefile,sep=""), 
          height=3.6,
          width=7,
          bg="transparent")
print(ggplt_smallmultivariate)
dev.off()

# ----------------
# For plotting
namefile <- paste("PC0_regressions_coefficients_",
                  "multivariate",
                  ".pdf", sep="")
cairo_pdf(paste(path_to_figures,namefile,sep=""), 
          height=3.6,
          width=7,
          bg="transparent")
print(ggplt_allmultivariate)
dev.off()




# -------------------------------------------------------


# =======================================================
# "SECRET WEAPON" plots
library(gridExtra)
# -------------------------------------------------------
predictor_vars <- c(allvars[1:3], "GovEffEstimate", "RegQualityEstimate", "RuleOfLawEstimate")
vars <- c("pc0_year", predictor_vars, "year")
ols_formula <- as.formula(paste(vars[1], paste(predictor_vars, collapse = " + "), sep = " ~ "))
regby_year <- na.omit(s_final_dt[, .SD, .SDcols = c("country_code", vars)]) %>% 
  group_by(year) %>%
  do(tidy(lm(formula = ols_formula, data = .), conf.int = .95)) %>%
  ungroup %>% rename(model = year) %>%
  arrange(model)
#regby_year$model <- factor(regby_year$model, levels = as.character(seq(1962, 2016, 1)))

ggplt_secretweapon_list <- list()
for(predvar in predictor_vars){
  ggplt_temp <- secret_weapon(regby_year, var = predvar) + 
    xlab(paste0("Std. Coefficient")) + ylab("Year") +
    ggtitle(paste0("Estimated Coefficient for ", predictors_names_new[predvar], " by year (Multivariate Regressions)")) +
    geom_vline(xintercept = 0, colour = "grey40", linetype = 4) +
    coord_flip() +
    scale_y_discrete(limits = sort(as.character(unique(regby_year$model)))) +
    scale_x_continuous(limits = c(-0.5, 0.85)) +
    theme_light() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 90, hjust = 1))
  
  ggplt_secretweapon_list[[predvar]] <- ggplt_temp
}

ggplt_secretweapon_list[2]


# ----------------
# For plotting
namefile <- paste("PC0_acrossyears_regressions_coefficients_",
                  "subset_multivariate",
                  ".pdf", sep="")
cairo_pdf(paste(path_to_figures,namefile,sep=""), 
          height=8,
          width=16,
          bg="transparent")
grid.arrange(grobs = ggplt_secretweapon_list, ncol = 2, nrow=3)
dev.off()




##############################################################
# Data Translation Project, Applied Econometrics
##############################################################
#3 Retail needs to worry about who has money to spend - what has changed about who is working and earning money?

#How to Answer the Question
##1. Cut-off of when COVID started (look into avg income industry)
  
# Load in Required Packages
library(vtable)
library(ipumsr)
library(tidyverse)
library(purrr)
library(lubridate)
library(fixest)
library(multcomp)
library(dplyr)
library(rdrobust)

# Read in and Load the Data
help("ipumsr")
ddi <- read_ipums_ddi("cps_00002.xml")
data <- read_ipums_micro(ddi)

# Step 1: look into the data (vtable) and choose the variables by using select(); rename any variables if needed. 
#then use sumtable to see the distribution
# View the Data
vtable(data)

# Clean and Use Select to choose required variables to make final data set
cleandt <- data %>% dplyr :: select(YEAR, MONTH, EMPSTAT, OCC, IND, UHRSWORK1, WKSTAT)
cleandt <- cleandt %>% drop_na()
cleandt <- cleandt %>% mutate(HRSMAIN = UHRSWORK1)

# We filter for Employment Status at 10 (employed) and 20 (unemployed)
cleandt <- cleandt %>% filter(EMPSTAT %in% c(10,20:22))

# Create a variable called date, that denotes for the different months in the years 2019 to 2022, use ggplot to visualize a change
#in Income Wage between 2019 to 2020
cleandt <- cleandt %>% mutate(date = ym(paste(YEAR,MONTH)))

# Join the Industry names and cleandt to align broader Industries
indnames <- read_csv('indnames.csv')
vtable(indnames)
cleandt <- cleandt %>% rename(ind = IND)
finaldt <- full_join(cleandt, indnames)
vtable(finaldt)

# Step 2: Visualize the difference in the count of employed and unemployed people between those of the different industries
finaldt$industry <- ifelse(finaldt$indname == 'Retail Trade', 1, 0)
emp_rate_by_month <- finaldt %>%
  group_by(industry, date) %>%
  summarize(emp_rate = 1 - mean(EMPSTAT %in% 20:22))
vtable(emp_rate_by_month)
sumtable(emp_rate_by_month)
emp_rate_by_month <- emp_rate_by_month %>% 
  drop_na(industry) %>%
  drop_na(date)
emp_rate_by_month %>% ggplot(aes(x = date, y = emp_rate, color = industry)) + geom_point() + labs(y = "Employed Rate", x = "Date") + 
  geom_vline(xintercept = as.Date("2020-03-01"))

# Step 3: Create bins of the Emp_Rate_By_Month
bindata <- emp_rate_by_month  %>% mutate(emp_bins = cut(emp_rate, breaks = 36))
sumtable(bindata)

# Step 4: use ggplot  -> visual representative of step 3
#View bindata in order to find the bin located at the intercept in which Covid Started which was 2020-03-01
view(bindata)
#Use the emp_bins designated for the 2020-03-01 date variable "0.952,0.955" to explain the cutoff
bindata %>% ggplot(aes(x = emp_bins)) + geom_bar() + labs(x = "Employed Rate Bins") + geom_vline(xintercept = '(0.952,0.955]') + theme(axis.text.x = element_blank())

# Step 5: rdrobust -> will select a window and weighting kernel to weight observations more near the cutoff
#mutate two new variables called pre-covid and post-covid to access the effect of covid on employment rate
treatment <- bindata %>% mutate(precovid = date < as.Date("2020-03-01"), postcovid = date > as.Date("2020-03-01"))

#use feols to look at a regression model to explain the difference
precovid <- feols(emp_rate ~ precovid, data = treatment)
etable(precovid, vcov = 'hetero')

postcovid <- feols(emp_rate ~ postcovid, data = treatment)
etable(precovid, postcovid, vcov = 'hetero')

#now create a variable and look at the effect covid had on employment rate during when vaccines did not exist
novacstreat <- treatment %>% filter(date < as.Date("2021-01-01"))
covidnovacs <- feols(emp_rate ~ postcovid, data = novacstreat)
etable(precovid, postcovid, covidnovacs)

#then create a variable and look at how the employment rate has improvement since the launch of vaccines
treatment <- treatment %>% mutate(postvacs = date > as.Date("2021-01-01"))
postvacs <- feols(emp_rate ~ postvacs, data = treatment)
etable(precovid, postcovid, covidnovacs, postvacs)

#we now want to look further at whether or not non-retail industries took a harder hit from Covid and benefited from vaccines
novacsind <- feols(emp_rate ~ postcovid*industry, data = novacstreat)
etable(covidnovacs, novacsind)
postvacsind <- feols(emp_rate ~ postvacs*industry, data = treatment)
etable(postvacs, postvacsind)

#Explanation of the Regression
## In postcovid, we are looking at how the overall impact of COVID-19 on employment rate. After march 2020, we see there is an 
## decrease in employment rate by 2.6%.In novacstreat, we are looking between when COVID-19 happen and before the vaccination 
## was released, to see the impact of COVID-19 employment which has a 5% decrease. In postvacs, we see that there is an overall 
## positive increase in terms of employment rate after vaccination was released on 2021, thus an increase by 2.2% on employment
## rate.

# Step 6: take a look at the ggplot again with linear lines explaining the effect
ggplot(treatment, aes(x = date, y = emp_rate, color = industry)) + geom_point() + labs(y = "Employment Rate", x = "Date") + 
  geom_vline(xintercept = as.Date('2020-03-01')) + theme(axis.text.x = element_blank()) + 
  geom_smooth(aes(group = precovid), method = 'lm', se = FALSE)


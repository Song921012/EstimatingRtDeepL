library(EpiNow2)
library(tidyverse)
library(lubridate)
reporting_delay <- estimate_delay(rlnorm(1000, log(2), 1),
                                  max_value = 15, bootstraps = 1)
reporting_delay <- list(
  mean = convert_to_logmean(2, 1), mean_sd = 0.1,
  sd = convert_to_logsd(2, 1), sd_sd = 0.1,
  max = 10
)
generation_time <- get_generation_time(disease = "SARS-CoV-2", source = "ganyani")
incubation_period <- get_incubation_period(disease = "SARS-CoV-2", source = "lauer")

reported_cases <- read_csv("E:\\Pythonfiles\\Github_Programs\\EstmatingRtDeepL\\Rt_Methods_Comparision\\ontario_cases.csv")
reported_cases$date <- ymd(reported_cases$date)
head(reported_cases)

estimates <- epinow(reported_cases = reported_cases,
                    generation_time = generation_time,
                    delays = delay_opts(incubation_period, reporting_delay),
                    rt = rt_opts(prior = list(mean = 2, sd = 0.2)),
                    stan = stan_opts(cores = 4))

names(estimates)

knitr::kable(summary(estimates))
Rt <- summary(estimates, type = "parameters", params = "R")

head(Rt)

write_csv(Rt, "ontario_epinow2_rt.csv")

It <- summary(estimates, output = "estimated_reported_cases")

head(It)

write_csv(It, "ontario_epinow2_It.csv")

plot(estimates)

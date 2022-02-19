library(EpiEstim)
library(tidyverse)
library(lubridate)




reported_cases <- read_csv("E:\\Pythonfiles\\Github_Programs\\EstmatingRtDeepL\\Rt_Methods_Comparision\\ontario_cases.csv")
reported_cases$date <- ymd(reported_cases$date)
head(reported_cases)

## make config
config_lit <- make_config(
  mean_si = 12.0,
  std_si = 5.2
)

epiestim_res_lit <- estimate_R(
  incid = reported_cases$confirm,
  method = "parametric_si",
  config = config_lit
)
ontario_epiestim_export <- as.data.frame(epiestim_res_lit$I_local)


write_csv(epiestim_res_lit$R, "ontario_epiestim_Rt.csv")
write_csv(ontario_epiestim_export, "ontario_epiestim_It.csv")


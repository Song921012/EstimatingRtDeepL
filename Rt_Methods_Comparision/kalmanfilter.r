library(EpiEstim)
library(tidyverse)
library(lubridate)
library(fpp2)

reported_cases <- read_csv("E:\\Pythonfiles\\Github_Programs\\EstmatingRtDeepL\\Rt_Methods_Comparision\\ontario_cases.csv")
reported_cases$date <- ymd(reported_cases$date)
head(reported_cases)

dailycase <- ma(reported_cases$confirm,5)
dailycase[1:2] = c(0.8,0.8)
dailycase[149:150] =c(166.6,166.6)

I <- c(0)
for (k in 1:(length(reported_cases$confirm)-1)){
  I[k+1] = 0.9*I[k] + dailycase[k]
}

Rt <- c(0)
for (k in 1:(length(reported_cases$confirm)-1)){
  Rt[k+1] = 10*(I[k+1]-I[k])/I[k] + 1
} 
Rt <- as.data.frame(Rt)

write_csv(Rt, "ontario_kalmanfilter_Rt.csv")

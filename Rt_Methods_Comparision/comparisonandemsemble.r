library(tidyverse)
library(lubridate)

reported_cases <- read_csv("E:\\Pythonfiles\\Github_Programs\\EstmatingRtDeepL\\Rt_Methods_Comparision\\ontario_cases.csv")
reported_cases$date <- ymd(reported_cases$date)
head(reported_cases)

home_path = "E:\\Pythonfiles\\Github_Programs\\EstmatingRtDeepL\\Rt_Methods_Comparision"
epinow2_Rt_data <- read_csv(file.path(home_path,"ontario_epinow2_Rt.csv"))
epinow2_It_data <- read_csv(file.path(home_path,"ontario_epinow2_It.csv"))
epiestime_Rt_data <- read_csv(file.path(home_path,"ontario_epiestim_Rt.csv"))
epiestime_It_data <- read_csv(file.path(home_path,"ontario_epiestim_It.csv"))
kalman_Rt_data <- read_csv(file.path(home_path,"ontario_kalmanfilter_Rt.csv"))
dl_data <- read_csv(file.path(home_path,"ontario_DL_export.csv"))

ontario <- tibble(date = reported_cases[7:143,"date"]$date)
ontario$case <- dl_data[7:143,"case"]$case
ontario$DeepLearningRt <- dl_data[7:143,"DLRt"]$DLRt
ontario$EpiEstimRt <- epiestime_Rt_data[7:143,]$`Mean(R)`
ontario$EpiNow2Rt <- epinow2_Rt_data[7:143,]$mean
ontario$KalmanRt <- kalman_Rt_data[7:143,]$Rt
ontario$DeepLearningIt <- dl_data[7:143,]$DLdaily
ontario$EpiEstimIt <- epiestime_It_data[7:143,]$`epiestim_res_lit$I_local`
ontario$EpiNow2It <- epinow2_It_data[7:143,]$mean
ontario$EmsembleRt <- (dl_data[7:143,"DLRt"]$DLRt + epiestime_Rt_data[7:143,]$`Mean(R)`+ epinow2_Rt_data[7:143,]$mean)/3
ontario$EmsembleIt <- (dl_data[7:143,]$DLdaily + epiestime_It_data[7:143,]$`epiestim_res_lit$I_local`+ epinow2_It_data[7:143,]$mean)/3
head(ontario)
write.csv(ontario,"DataSum.csv")

ggplot(ontario, aes(x = 1:length(date), y = case)) + geom_point()+labs(x = "Days after March 1", y = "Daily Cases")

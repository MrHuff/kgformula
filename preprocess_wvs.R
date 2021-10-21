WVS_Trend_v2_0 <- readRDS("~/Downloads/F00011411_WVS_Trend_1981_2020_R_rds_v2_0/WVS_Trend_v2_0.rds")

columns_of_interest = c("X003","A170","C006","X011","X047_WVS","X007","X023","X028")
rename_cols = c("age", "satisfaction_life", "financial_happiness", "number_kids","income","martial_status","age_education_complete","employment_status")

subset <-WVS_Trend_v2_0[,columns_of_interest]
names(subset)<-rename_cols
# subset <- subset[subset>=0]


setwd("~/Documents/phd_projects/kgformula")
write.csv(subset,"wvs.csv")



library(hdm)
library(ggplot2)
setwd("~/Documents/phd_projects/kgformula")
treatments =c(
  'School closing',
  'Workplace closing',
  'Cancel public events',
  'Restrictions on gatherings',
  'Close public transport',
  'Masks'
  )
n=2500
Y <-as.matrix(read.csv(file = "covid_19_1/covid_Y.csv",header = TRUE))
Z_cont <-as.matrix(read.csv(file = "covid_19_1/covid_Z_cont.csv",header = TRUE))
for (treatment_string in treatments){
  df = as.data.frame(matrix(nrow = 1, ncol = 100))
  names(df)<-c(1:100)
  j=1
  vec<- double(100)
  t_file = sprintf("covid_19_1/covid_T=%s.csv", treatment_string)
  z_cat_file = sprintf("covid_19_1/covid_Z_cat=%s.csv",treatment_string)
  save_file = sprintf("~/Documents/phd_projects/kgformula/covid_bench_T=%s_n=%s.csv",treatment_string , n)
  T <-as.matrix(read.csv(file = t_file,header = TRUE))
  Z_cat <-as.matrix(read.csv(file = z_cat_file,header = TRUE))
  for (i in 0:99){
    set.seed(i)
    mask = sample(c(1:nrow(T)), size=2500, replace =F)
    
    x = T[mask,]
    y = Y[mask] 
    z_1 = Z_cont[mask,]
    z_2 = Z_cat[mask,]
    data <- cbind(x,z_1,z_2)
    mod = rlassoEffects(data, y, index=c(1,2,3,4,5,6,7,8),joint=TRUE)
    vec[i+1]=mod$pval
  }
  df[j,] = vec
  write.csv(df,save_file, row.names = FALSE)
  hist(vec)
  
  
  
}



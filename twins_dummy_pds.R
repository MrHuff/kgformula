library(hdm)
library(ggplot2)
setwd("~/Documents/phd_projects/kgformula")
df = as.data.frame(matrix(nrow = 1, ncol = 100))
names(df)<-c(1:100)
j=1

vec<- double(100)
T <-as.matrix(read.csv(file = "twins_T.csv",header = TRUE))
Y <-as.matrix(read.csv(file = "twins_Y.csv",header = TRUE))
Z_cont <-as.matrix(read.csv(file = "twins_z_cont.csv",header = TRUE))
Z_cat <-as.matrix(read.csv(file = "twins_z_cat.csv",header = TRUE))

n=5000

for (i in 0:99){
  set.seed(i)
  mask = sample(c(1:nrow(T)), size=n, replace =F)
  
  x = T[mask]
  y = rnorm(n) 
  z_1 = Z_cont[mask,]
  z_2 = Z_cat[mask,]
  data <- cbind(x,z_1,z_2)
  mod = rlassoEffects(data, y, index=c(1,2,3))
  vec[i+1]=mod$pval
}
df[j,] = vec

write.csv(df,"~/Documents/phd_projects/kgformula/twins_bench_1d_5000_dummy.csv", row.names = FALSE)



hist(vec)




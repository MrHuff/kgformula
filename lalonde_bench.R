library(hdm)
library(ggplot2)
setwd("~/Documents/phd_projects/kgformula")
df = as.data.frame(matrix(nrow = 1, ncol = 100))
names(df)<-c(1:100)
j=1
fname = "lalonde.csv"
vec<- double(100)
dat <-as.matrix(read.csv(file = fname,header = TRUE))
X <-  dat[,"treat"]
Y <-  dat[,"re78"]
Z <-   dat[,c('age','education','black','hispanic','married','nodegree','re74','re75')]
for (i in 0:99){
  set.seed(i)
  mask = sample(c(1:nrow(dat)), size=100, replace =F)
  
  x = X[mask]
  y = Y[mask] 
  z = Z[mask,]
  data <- cbind(x,z)
  mod = rlassoEffects(data, y, index=c(1))
  vec[i+1]=mod$pval
}
df[j,] = vec

write.csv(df,"~/Documents/phd_projects/kgformula/lalonde_bench_1d.csv", row.names = FALSE)
hist(vec)



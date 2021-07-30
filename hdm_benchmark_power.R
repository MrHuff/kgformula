library(hdm)
library(ggplot2)
setwd("~/Documents/phd_projects/kgformula")
alphas = list("0.0",0.02,0.04,0.06,0.08,0.1)
Ns = list(1000,5000,10000)
d = list(1)
null_case = list("True","False")
df = as.data.frame(matrix(nrow = 1, ncol = 104))
names(df)<-c("alp","null","d","n",c(1:100))
j=1
for (alp in alphas){
  for (null in null_case){
    for (dim in d){
      fname = sprintf("do_null_univariate_alp=%s_null=%s_d=%s_csv",alp,null,dim)
      for (n in Ns){
        vec<- double(100)
        for (i in 0:99){
          x <-  as.matrix(read.csv(file = sprintf("%s/x_%s.csv",fname,i),header = FALSE))
          y <-  as.matrix(read.csv(file = sprintf("%s/y_%s.csv",fname,i),header = FALSE))
          z <-  as.matrix(read.csv(file = sprintf("%s/z_%s.csv",fname,i),header = FALSE))
          x = head(x, n)
          y =  head(y, n)
          z = head(z, n)
          data <- cbind(x,z)
          mod = rlassoEffects(data, y[,1], index=c(1))
          #  mod = rlassoEffect(z, y, x, method = "double selection", I3 = NULL, post = TRUE)
          vec[i+1]=mod$pval
        }
        df[j,] = c(alp,null,dim,n,vec)
        j=j+1
        
      }
     
    }
  }
}

write.csv(df,"~/Documents/phd_projects/kgformula/hdm_bench_syntehtic.csv", row.names = FALSE)




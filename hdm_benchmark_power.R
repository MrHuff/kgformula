library(hdm)
library(ggplot2)
setwd("~/Documents/phd_projects/kgformula")
alphas = list("0.0",0.001,0.002,0.003,0.004,0.005,0.008,0.012,0.016,0.02)
Ns = list(1000,5000,10000)
d = list(1)
null_case = list("False")
df = as.data.frame(matrix(nrow = 1, ncol = 104))
names(df)<-c("alp","null","d","n",c(1:100))
j=1
dim=1
for (alp in alphas){
    fname = sprintf("do_null_100_csv/beta_xy=[0, %s]_d_X=1_d_Y=1_d_Z=1_n=10000_yz=[0.5, 0.0]_beta_XZ=0.25_theta=2.0_phi=2.0",alp)
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
      df[j,] = c(alp,F,dim,n,vec)
      j=j+1
      
    }
}

write.csv(df,"~/Documents/phd_projects/kgformula/hdm_bench_syntehtic_cont.csv", row.names = FALSE)




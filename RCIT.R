setwd("~/Documents/phd_projects/kgformula")

library(devtools)
library(RCIT)


Ns = list(1000,5000,10000)
df = as.data.frame(matrix(nrow = 1, ncol = 101))
df_2 = as.data.frame(matrix(nrow = 1, ncol = 101))

names(df)<-c("n",c(1:100))
names(df_2)<-c("n",c(1:100))

j=1
for (n in Ns){
  vec<- double(100)
  vec_2<- double(100)
  for (i in 0:99){
    load_x = sprintf("exp_gcm_break_100/beta_xy=[0.0, 0.0]_d_X=1_d_Y=1_d_Z=1_n=10000_yz=[-0.5, 4.0]_beta_XZ=0.0_theta=1.0_phi=2.0_csv/x_%s.csv", i)
    load_y = sprintf("exp_gcm_break_100/beta_xy=[0.0, 0.0]_d_X=1_d_Y=1_d_Z=1_n=10000_yz=[-0.5, 4.0]_beta_XZ=0.0_theta=1.0_phi=2.0_csv/y_%s.csv", i)
    load_z = sprintf("exp_gcm_break_100/beta_xy=[0.0, 0.0]_d_X=1_d_Y=1_d_Z=1_n=10000_yz=[-0.5, 4.0]_beta_XZ=0.0_theta=1.0_phi=2.0_csv/z_%s.csv", i)
    x = as.matrix(read.csv(load_x,header=FALSE))
    y = as.matrix(read.csv(load_y,header=FALSE))
    z = as.matrix(read.csv(load_z,header=FALSE))
    x = head(x, n)
    y =  head(y, n)
    z = head(z, n)
    
    o_1 = RCIT(x,y,z)
    o_2 = RCoT(x,y,z)
    vec[i+1]=o_1$p
    vec_2[i+1]=o_2$p
    
    #output = gcm.test(x,y,z)
    #vec[i+1]=output$p.value
    
    # output = CondIndTest(X, Y, Z, method = "ResidualPredictionTest")
    # vec[i+1]=output$pvalue
    
    # output = CondIndTest(X, Y, Z, method = "KCI")
    # vec[i+1]=output$pvalue
    
  }
  df[j,] = c(n,vec)
  df_2[j,]=c(n,vec_2)
  j=j+1
}


write.csv(df,"~/Documents/phd_projects/kgformula/rcit.csv", row.names = FALSE)
write.csv(df_2,"~/Documents/phd_projects/kgformula/rcot.csv", row.names = FALSE)








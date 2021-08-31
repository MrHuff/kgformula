library(GeneralisedCovarianceMeasure)
setwd("~/Documents/phd_projects/kgformula")

Ns = list(1000,5000,10000)
df = as.data.frame(matrix(nrow = 1, ncol = 101))
names(df)<-c("n",c(1:100))
j=1
for (n in Ns){
  vec<- double(100)
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
    output = gcm.test(x,y,z)
    vec[i+1]=output$p.value
  }
  df[j,] = c(n,vec)
  j=j+1
}
write.csv(df,"~/Documents/phd_projects/kgformula/gcm_break_ref.csv", row.names = FALSE)



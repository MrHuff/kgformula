library(GeneralisedCovarianceMeasure)
library(CondIndTests)
source('parCopCITest.R')

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
    
    output <- test_CI(X=x, Y=y, Z=z,
                      quantile_reg = 'B-Spline',
                      bspline_df = 3, 
                      poly_deg = 3,
                      q = c(5), 
                      tau_min = 0.01,
                      tau_max = 0.99, 
                      delta = 0.01)
    vec[i+1]=output$p_value
    
    #output = gcm.test(x,y,z)
    #vec[i+1]=output$p.value
    
    # output = CondIndTest(X, Y, Z, method = "ResidualPredictionTest")
    # vec[i+1]=output$pvalue
    
    # output = CondIndTest(X, Y, Z, method = "KCI")
    # vec[i+1]=output$pvalue
    
  }
  df[j,] = c(n,vec)
  j=j+1
}


write.csv(df,"~/Documents/phd_projects/kgformula/quantile_jmlr_break_ref.csv", row.names = FALSE)



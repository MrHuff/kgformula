library(hdm)
library(ggplot2)
setwd("~/Documents/phd_projects/kgformula")

n_list = c(1000,5000,10000)
fold  = "hdm_breaker_fam_y=4_100_csv"
y_style=4
xy_list = c("0.0",0.001,0.002,0.003,0.004,0.005)
for (beta_xy in xy_list){
  for (n in n_list){
    j=1
    df = as.data.frame(matrix(nrow = 1, ncol = 100))
    names(df)<-c(1:100)
    vec<- double(100)
    save_file = sprintf("~/Documents/phd_projects/kgformula/hdm_fail_cont_xy=%s_n=%s_y=%s.csv",beta_xy , n,y_style)
    for (i in 0:99){
      t_file = sprintf("%s/beta_xy=[0, %s]_d_X=3_d_Y=3_d_Z=50_n=10000_yz=[0.5, 0.0]_beta_XZ=0.075_theta=16.0_phi=2.0/x_%s.csv",fold, beta_xy,i)
      y_file = sprintf("%s/beta_xy=[0, %s]_d_X=3_d_Y=3_d_Z=50_n=10000_yz=[0.5, 0.0]_beta_XZ=0.075_theta=16.0_phi=2.0/y_%s.csv",fold,beta_xy,i)
      z_file = sprintf("%s/beta_xy=[0, %s]_d_X=3_d_Y=3_d_Z=50_n=10000_yz=[0.5, 0.0]_beta_XZ=0.075_theta=16.0_phi=2.0/z_%s.csv",fold,beta_xy,i)
      
      T <-as.matrix(read.csv(file = t_file,header = TRUE))
      Y <-as.matrix(read.csv(file = y_file,header = TRUE))
      Z_cont <-as.matrix(read.csv(file = z_file,header = TRUE))
      
      set.seed(i)
      if (n<10000){
        mask = sample(c(1:nrow(T)), size=n, replace =F)
        
      }else{
        mask =c(1:nrow(T))
      }
      
      x = T[mask,]
      y = Y[mask] 
      z_1 = Z_cont[mask,]
      data <- cbind(x,z_1)
      mod = rlassoEffects(data, y, index=c(1),joint=TRUE)
      vec[i+1]=mod$pval
    }
    df[j,] = vec
    write.csv(df,save_file, row.names = FALSE)
    hist(vec)
  }  
  
}




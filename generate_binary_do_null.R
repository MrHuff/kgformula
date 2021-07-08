library(purrr)
library(causl)

setwd("~/Documents/phd_projects/kgformula")
source("include_weights_sample.R")




forms <- list(Z ~ 1,
              list(X ~ Z),  ## this doesn’t have to be a list, but it can be if you want
              Y ~ X, 
              ~ X
)


pars <- list(
  X = list(beta = c(0.3,-0.2)),   ## doesn’t need a dispersion, since it’s binomial
  Z = list(beta = 0, phi=1),     ## needs a dispersion (phi), since it’s Gamma
  Y = list(beta = c(-0.5,0.2), phi=1), #Change 0.2 to 0 for null
  cop = list(beta = c(1,0.5))) # controls dependency between x and z. 

fold_name="do_null_binary_csv_debug"
if (!dir.exists(fold_name)){
  dir.create(fold_name)
}

for (i in 0:99) {
  set.seed(i)
  dat_max <- causalSamp(1e4, formulas = forms, pars=pars, family = list(3,5,1,1))
  
  options(digits=3)
  ##summary(glm(X ~ 1, family=binomial, data=dat_max))$coef
  #summary(glm(L ~ A0, family=Gamma(link="log"), data=dat_max))$coef
  glmA1 <- glm(X ~ Z, family=binomial, data=dat_max)
  #summary(glmA1)$coef
  
  
  w <- predict(glmA1, type="response")
  w[dat_max$X == 0] <- 1 - w[dat_max$X == 0]
  #summary(lm(Y ~ A0*A1, data=dat_max, weights = 1/w))$coef
  
  name = sprintf("%s/data_seed=%s.csv",fold_name,i)

  write.csv(dat_max,name,row.names = FALSE)
}

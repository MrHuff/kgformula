library(purrr)
library(causl)

forms <- list(L ~ A0, 
              list(A0 ~ 1, A1 ~ A0*L),
              Y ~ A0*A1, 
              ~ A0)

pars <- list(A0 = list(beta = 0),
             L = list(beta = c(0.3,-0.2), phi=1),
             A1 = list(beta = c(-0.3,0.4,0.3,0)), 
             Y = list(beta = c(-0.5,0.2,0.3,0), phi=1),
             cop = list(beta = c(1,0.5)))

dir.create("do_null_binary_csv")

for (i in 0:99) {
  set.seed(i)
  dat_max <- causalSamp(1e4, formulas = forms, pars=pars, family = list(3,c(5,5),1,1))
  
  options(digits=3)
  summary(glm(A0 ~ 1, family=binomial, data=dat_max))$coef
  summary(glm(L ~ A0, family=Gamma(link="log"), data=dat_max))$coef
  glmA1 <- glm(A1 ~ A0*L, family=binomial, data=dat_max)
  summary(glmA1)$coef
  
  
  w <- predict(glmA1, type="response")
  w[dat_max$A1 == 0] <- 1 - w[dat_max$A1 == 0]
  summary(lm(Y ~ A0*A1, data=dat_max, weights = 1/w))$coef
  
  name = sprintf("do_null_binary_csv/data_seed=%s.csv", i)

  write.csv(dat_max,name,row.names = FALSE)
}

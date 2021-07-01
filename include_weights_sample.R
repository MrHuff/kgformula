library(purrr)
library(causl)

rejectionWeights <- function (dat, mms,# formula,
                              family, pars, qden) {
  
  d <- ncol(dat)
  
  if (d != length(mms)) stop("Inconsistent length of dat and mms")
  if (d != length(family)) stop("Inconsistent length of dat and family")
  if (d != length(pars)) stop("Inconsistent length of dat and pars")
  if (d != length(qden)) stop("Inconsistent length of dat and family")
  
  betas <- lapply(pars, function(x) x$beta)
  eta <- mapply(function(X,y) X %*% y, mms, betas, SIMPLIFY = FALSE)
  
  wts <- rep(1, nrow(dat))
  
  for (i in seq_len(d)) {
    phi <- pars[[i]]$phi
    
    if (family[i] == 1) {
      mu <- eta[[i]]
      wts <- wts*dnorm(dat[,i], mean=mu, sd=sqrt(phi))/qden[[i]](dat[,i])
    }
    else if (family[i] == 2) {
      mu <- eta[[i]]
      wts <- wts*dt((dat[,i] - mu)/sqrt(phi), df=pars[[i]]$par2)/(sqrt(phi)*qden[[i]](dat[,i]))
    }
    else if (family[i] == 3) {
      mu <- exp(eta[[i]])
      wts <- wts*dgamma(dat[,i], rate=1/(mu*phi), shape=1/phi)/qden[[i]](dat[,i])
    }
    else if (family[i] == 4) {
      mu <- expit(eta[[i]])
      wts <- wts*dbeta(dat[,i], shape1=1+phi*mu, shape2=1+phi*(1-mu))/qden[[i]](dat[,i])
    }
    else if (family[i] == 5) {
      mu <- expit(eta[[i]])
      wts <- wts*dbinom(dat[,i], prob=mu, size=1)/qden[[i]](dat[,i])
    }
    else stop("family[2] must be 1, 2, 3 or 4")
  }
  
  if (any(is.na(wts))) stop("Problem with weights")
  
  wts
}

lhs <- function (formulas) {
  if (!is.list(formulas)) formulas <- list(formulas)
  term <- lapply(formulas, terms)
  
  ## get list of variables
  vars <- lapply(term, function(x) attr(x, which="variables"))
  resp <- rep(NA_character_, length(vars)) #character(length(vars))
  if (any(lengths(vars) >= 2)) {
    mis <- (sapply(term, attr, which="response") != 1)
    if (!all(mis)) resp[!mis & lengths(vars) >= 2] <- sapply(vars[!mis& lengths(vars) >= 2], function(x) as.character(x[[2]]))
  }
  # mis <- (sapply(term, attr, which="response") != 1)
  # resp[mis] <- rep()
  
  resp
}


rhs_vars <- function (formulas) {
  if (!is.list(formulas)) formulas <- list(formulas)
  term <- lapply(formulas, terms)
  wres <- (sapply(term, attr, which="response") == 1)
  nores <- (sapply(term, attr, which="response") == 0)
  
  vars <- lapply(term, function(x) attr(x, which="variables"))
  
  ## get list of variables
  vars[wres] <- lapply(vars[wres], function(x) as.character(x[-(1:2)]))
  vars[nores] <- lapply(vars[nores], function(x) as.character(x[-1]))
  
  # if (any(lengths(vars) >= 2)) resp[lengths(vars) >= 3] <- lapply(vars[lengths(vars) >= 3], function(x) as.character(x))
  
  # resp[mis] <- rep()
  
  vars
}

rescaleVar <- function(U, X, pars, family=1, link) {
  
  ## get linear component
  eta <- X %*% pars$beta
  phi <- pars$phi
  
  ## make U normal, t or gamma
  if (family == 1) {
    Y <- qnorm(U, mean = eta, sd=sqrt(phi))
  }
  else if (family == 2) {
    Y <- sqrt(phi)*qt(U, df=pars$par2) + eta
  }
  else if (family == 3) {
    Y <- qexp(U, rate = 1/(exp(eta)*sqrt(phi)))
  }
  else if (family == 4) {
    Y <- qbeta(U, shape1 = 1, shape2 = 1)
  }
  else if (family == 5) {
    trunc <- pars$trunc
    trnc <- 1
    
    stop("Not finished family==5 yet")
    mat <- matrix(NA, length(U), length(trunc))
    for (j in seq_along(trunc[[trnc]])) {
      mat[,j] <- 1*(U > trunc[[trnc]][j])
    }
    Y <- rowSums(mat)
  }
  else stop("family must be 1, 2 or 3")
  
  ### get Z values to correct families
  # nms <- names(dat)[grep("z", names(dat))]
  
  return(Y)
}


causalSamp_2 <- function(n, formulas = list(list(z ~ 1), list(x ~ z), list(y ~ x), list( ~ 1)),
                         pars, family, link=NULL, dat=NULL,
                         control=list(), seed) {
  
  # get control parameters or use defaults
  con = list(oversamp = 10, max_oversamp=1000, max_wt = 1, warn = 1)
  matches = match(names(control), names(con))
  con[matches] = control[!is.na(matches)]
  if (any(is.na(matches))) warning("Some names in control not matched: ",
                                   paste(names(control[is.na(matches)]),
                                         sep = ", "))
  if (round(con$oversamp) != con$oversamp) {
    con$oversamp <- ceiling(con$oversamp)
    message("oversamp not an integer, rounding up")
  }
  if (round(con$max_oversamp) != con$max_oversamp) {
    con$max_oversamp <- ceiling(con$max_oversamp)
    message("max_oversamp not an integer, rounding up")
  }
  
  ## check we have four groups of formulas
  if (length(formulas) != 4) stop("formulas must have length 4")
  if (missing(pars)) stop("Must supply parameter values")
  
  ## ensure all formulas are given as lists
  if (any(sapply(formulas, class) == "formula")) {
    wh <- which(sapply(formulas, class) == "formula")
    for (i in wh) formulas[[i]] <- list(formulas[[i]])
  }
  
  datNULL <- is.null(dat)
  dim <- lengths(formulas[1:3])
  
  if (missing(family)) {
    family = lapply(lengths(formulas), function(x) rep.int(1,x))
  }
  
  ## set seed to 'seed'
  if (missing(seed)) {
    seed <- round(1e9*runif(1))
  }
  # else .Random.seed <- seed
  set.seed(seed)
  
  if (all(unlist(family) == 0)) {
    message("Perhaps better to simulate this using the MLLPs package")
  }
  # else if (family[1] == 0 && family[3] == 0 && family[4] != 0) {
  #   warning("discrete data does not work well with continuous copulas")
  # }
  
  dZ <- dim[1]
  dX <- dim[2]
  dY <- dim[3]
  
  famZ <- family[[1]]
  famX <- family[[2]]
  famY <- family[[3]]
  famCop <- family[[4]]
  
  ## check variable names
  LHS_Z <- lhs(formulas[[1]])
  LHS_X <- lhs(formulas[[2]])
  LHS_Y <- lhs(formulas[[3]])
  output <- c(LHS_Z, LHS_Y)
  vars <- c(LHS_Z, LHS_X, LHS_Y)
  if (!datNULL) {
    vars <- c(names(dat), vars)
  }
  if (anyDuplicated(na.omit(vars))) stop("duplicated variable names")
  
  # ## get names for three main variables
  # nmZ <- LHS[1]
  # nmX <- LHS[2]
  # nmY <- LHS[3]
  # pars2 <- pars
  # names(pars2)[match(LHS[1:3], names(pars2))] <- c("z", "x", "y")
  
  ## set up data frame for output
  oversamp <- con$oversamp
  
  if (!datNULL) {
    dat2 <- dat[rep(nrow(dat), each=oversamp), ]
  }
  
  out <- data.frame(matrix(0, ncol=sum(dim), nrow=n*oversamp))
  names(out) <- c(LHS_Z, LHS_X, LHS_Y)
  
  if (!datNULL) {
    out <- cbind(dat2, out)
  }
  
  ## list for densities used to simulate X's
  qden <- vector(mode="list", length=dim[2])
  
  for (i in seq_len(dX)) {
    ## get parameters for X
    if (famX[i] == 0 || famX[i] == 5) {
      if (!is.null(pars[[LHS_X[i]]]$p)) theta <- pars[[LHS_X[i]]]$p
      else theta <- 0.5
      famX[i] <- 5
    }
    else if (famX[i] == 1 || famX[i] == 2) {
      theta <- 2*pars[[LHS_X[i]]]$phi
    }
    else if (famX[i] == 3) {
      theta <- 2*pars[[LHS_X[i]]]$phi
    }
    else if (famX[i] == 4) {
      theta = c(1,1)
    }
    
    ## obtain data for X's
    tmp <- sim_X(n*oversamp, fam_x = famX[i], theta=theta)
    out[LHS_X[[i]]] <- tmp$x
    qden[[i]] <- tmp$qden
  }
  # ## give default coefficients
  # if (is.null(pars2$z$beta)) pars2$z$beta = 0
  # if (is.null(pars2$z$phi)) pars2$z$phi = 1
  
  ## get linear predictors
  mms <- vector(mode = "list", length=3)
  mms[c(1,3)] = rapply(formulas[c(1,3)], model.matrix, data=out, how = "list")
  for (i in seq_along(mms[[1]])) {
    if (ncol(mms[[1]][[i]]) != length(pars[[LHS_Z[i]]]$beta)) stop(paste0("dimension of model matrix for ", LHS_Z[i], " does not match number of coefficients provided"))
  }
  for (i in seq_along(mms[[3]])) {
    if (ncol(mms[[3]][[i]]) != length(pars[[LHS_Y[i]]]$beta)) stop(paste0("dimension of model matrix for ", LHS_Y[i], " does not match number of coefficients provided"))
  }
  
  # etas <- vector(mode="list", length=3)
  # for (i in c(1,3)) {
  #   etas[[i]] <- mapply(function(x, y) x %*% pars[[y]]$beta, mms[[i]], lhs(formulas[[i]]), SIMPLIFY = FALSE)
  # }
  copMM <- model.matrix(formulas[[4]][[1]], out)
  if (is.matrix(pars$cop$beta) && (nrow(pars$cop$beta) != ncol(copMM))) stop(paste0("dimension of model matrix for copula (", ncol(copMM), ") does not match number of coefficients provided (", nrow(pars$cop$beta),")"))
  
  # eta <- list()
  # eta$z <- mms[[1]] %*% pars2$z$beta
  # eta$y <- mms[[2]] %*% pars2$y$beta
  # mms[[3]] <- model.matrix(update.formula(formulas[[4]], NULL ~ . ), out)
  
  ## code to check if Y or Z is included in copula formula
  if (any(output %in% rhs_vars(formulas[[4]])[[1]])) stop("copula cannot depend upon Z or Y variables")
  if (length(famCop) > 1) {
    if (nrow(unique(copMM)) > 25) warning("using vine copulas with continuous covariates may be very slow")
  }
  ## get copula data and then modify distributions of Y and Z
  out[,output] <- sim_CopVal(out[,output], family=famCop,
                             par = pars$cop, par2=pars$cop$par2, model_matrix=copMM)
  for (i in seq_along(LHS_Z)) out[[LHS_Z[i]]] <- rescaleVar(out[[LHS_Z[i]]], X=mms[[1]][[i]],
                                                            family=famZ[i], pars=pars[[LHS_Z[i]]])
  for (i in seq_along(LHS_Y)) out[[LHS_Y[i]]] <- rescaleVar(out[[LHS_Y[i]]], X=mms[[3]][[i]],
                                                            family=famY[i], pars=pars[[LHS_Y[i]]])
  
  mms[[2]] = lapply(formulas[[2]], model.matrix, data=out)
  
  ## perform rejection sampling
  wts_ref <- rejectionWeights(out[LHS_X], mms[[2]], family=famX, pars=pars[LHS_X], qden = qden)
  con$max_wt <- max(max(wts_ref), con$max_wt)
  wts <- wts_ref/con$max_wt
  # if (mean(wts > 0.2) < 0.01) {
  #   if (con$warn == 1) warning("weights may be unbounded")
  #   else if (con$warn == 2) stop("weights may be unbounded")
  # }
  
  ## different behaviour depending upon whether covariates were supplied
  if (!datNULL) {
    return(out)
    stop("Haven't done this bit yet!")
    # done <- matrix(runif(nrow(out)) < wts, nrow=oversamp)
    # wh2 <- apply(done, 2, function(x) which(x)[1])
    # nr <- sum(colSums(done) > 0)
  }
  else {
    bool_mask <- runif(nrow(out)) < wts
    wts_ref <- wts_ref[bool_mask]
    out2 <- out[bool_mask, ]
    out2$wts = 1/wts_ref
    nr2 <- nrow(out2)
    
    if (nr2 < n) {
      ## not enough rows, so consider increasing oversampling rate
      if (con$oversamp == con$max_oversamp) {
        ## already at maximum oversample, so just return what we have
        warning(paste0("Only sampled ", nr2, " observations"))
      }
      else {
        con$oversamp <- min(con$max_oversamp, ceiling(con$oversamp*(n/nr2*1.1)))
        out2 <- Recall(n, formulas, pars=pars, family=family, link=link, dat=dat, control=con, seed=seed)
      }
    }
    else if (nr2 > n) {
      ## too many rows, so take a subsample
      out2 <- out2[seq_len(n), ]
    }
  }
  
  rownames(out2) <- NULL
  
  return(out2)
}

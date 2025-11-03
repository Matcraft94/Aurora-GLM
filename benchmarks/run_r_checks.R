#!/usr/bin/env Rscript
# R script to validate Aurora-GLM against R's glm() function
# Usage: Rscript benchmarks/run_r_checks.R [--output results.json]

library(jsonlite)

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
output_file <- if ("--output" %in% args) args[which(args == "--output") + 1] else NULL

# Set seed for reproducibility
set.seed(42)

#' Run a single GLM comparison
#'
#' @param family_name Character name of GLM family ("gaussian", "poisson", "binomial", "Gamma")
#' @param link_name Character name of link function
#' @param n_obs Integer number of observations
#' @param n_features Integer number of features (excluding intercept)
#' @param replicate Integer replicate number
#' @return Named list with comparison results
run_comparison <- function(family_name, link_name, n_obs = 200, n_features = 3, replicate = 1) {
  # Generate synthetic data
  X <- matrix(rnorm(n_obs * n_features, mean = 0, sd = 1), nrow = n_obs, ncol = n_features)
  colnames(X) <- paste0("X", seq_len(n_features))

  coef_true <- rnorm(n_features, mean = 0, sd = 0.7)
  intercept_true <- rnorm(1, mean = 0, sd = 0.5)
  linear_predictor <- intercept_true + X %*% coef_true

  # Generate response based on family
  if (family_name == "gaussian") {
    y <- as.vector(linear_predictor + rnorm(n_obs, mean = 0, sd = 0.5))
    family_obj <- gaussian(link = link_name)
  } else if (family_name == "poisson") {
    mu <- exp(linear_predictor)
    y <- rpois(n_obs, lambda = mu)
    family_obj <- poisson(link = link_name)
  } else if (family_name == "binomial") {
    probs <- 1 / (1 + exp(-linear_predictor))
    y <- rbinom(n_obs, size = 1, prob = probs)
    family_obj <- binomial(link = link_name)
  } else if (family_name == "Gamma") {
    shape <- 2.0
    mu <- exp(pmin(pmax(linear_predictor, -3), 3))  # Clip for stability
    scale <- mu / shape
    y <- rgamma(n_obs, shape = shape, scale = scale)
    family_obj <- Gamma(link = link_name)
  } else {
    stop(paste("Unsupported family:", family_name))
  }

  # Fit GLM in R
  df <- data.frame(y = y, X)
  formula <- as.formula(paste("y ~", paste(colnames(X), collapse = " + ")))

  tryCatch({
    fit <- glm(formula, data = df, family = family_obj, control = list(maxit = 100))

    # Extract results
    list(
      family = family_name,
      link = link_name,
      replicate = replicate,
      n_obs = n_obs,
      n_features = n_features,
      intercept = unname(coef(fit)[1]),
      coefficients = unname(coef(fit)[-1]),
      deviance = deviance(fit),
      null_deviance = fit$null.deviance,
      aic = AIC(fit),
      converged = fit$converged,
      n_iter = fit$iter,
      fitted_values = unname(fitted(fit)),
      # Save data for Python comparison
      X = X,
      y = y,
      success = TRUE,
      error = NA_character_
    )
  }, error = function(e) {
    list(
      family = family_name,
      link = link_name,
      replicate = replicate,
      n_obs = n_obs,
      n_features = n_features,
      success = FALSE,
      error = conditionMessage(e)
    )
  })
}

#' Run all comparisons
run_all_comparisons <- function(n_replicates = 5) {
  configs <- list(
    list(family = "gaussian", link = "identity"),
    list(family = "poisson", link = "log"),
    list(family = "binomial", link = "logit"),
    list(family = "Gamma", link = "log")
  )

  results <- list()

  cat("Running R GLM validation checks...\n")
  cat(sprintf("%-10s | %-10s | %-5s | %-10s | %-10s | %-10s\n",
              "Family", "Link", "Rep", "Converged", "Deviance", "AIC"))
  cat(strrep("-", 70), "\n")

  for (config in configs) {
    for (rep in seq_len(n_replicates)) {
      result <- run_comparison(
        family_name = config$family,
        link_name = config$link,
        n_obs = 200,
        n_features = 3,
        replicate = rep
      )

      results <- c(results, list(result))

      if (result$success) {
        cat(sprintf("%-10s | %-10s | %5d | %-10s | %10.4f | %10.4f\n",
                    result$family,
                    result$link,
                    result$replicate,
                    ifelse(result$converged, "Yes", "No"),
                    result$deviance,
                    result$aic))
      } else {
        cat(sprintf("%-10s | %-10s | %5d | FAILED: %s\n",
                    result$family,
                    result$link,
                    result$replicate,
                    result$error))
      }
    }
  }

  cat(strrep("-", 70), "\n")
  cat(sprintf("Completed %d comparisons\n", length(results)))

  results
}

# Run comparisons
results <- run_all_comparisons(n_replicates = 5)

# Write output if requested
if (!is.null(output_file)) {
  # Convert matrices/arrays to lists for JSON serialization
  results_json <- lapply(results, function(r) {
    if (!is.null(r$X)) r$X <- unname(as.vector(t(r$X)))  # Flatten row-major
    if (!is.null(r$y)) r$y <- unname(as.vector(r$y))
    if (!is.null(r$fitted_values)) r$fitted_values <- unname(as.vector(r$fitted_values))
    if (!is.null(r$coefficients)) r$coefficients <- unname(as.vector(r$coefficients))
    r
  })

  json_output <- toJSON(results_json, pretty = TRUE, auto_unbox = TRUE, na = "null")
  write(json_output, file = output_file)
  cat(sprintf("\nWrote results to %s\n", output_file))
}

# Return exit code based on success
n_failures <- sum(sapply(results, function(r) !r$success))
if (n_failures > 0) {
  quit(status = 1)
} else {
  quit(status = 0)
}

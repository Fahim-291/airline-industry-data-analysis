
# ── Load & Sample Data
setwd("/Resources/")
df_full <- read.csv("flight_data.csv", header = TRUE, stringsAsFactors = FALSE)


# ── Load packages
library(ggplot2)
library(dplyr)
library(corrplot)
library(reshape2)
library(psych)
library(moments)

ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point()
mtcars %>%
  filter(mpg > 20) %>%
  select(mpg, hp)
corr_matrix <- cor(mtcars)
corrplot(corr_matrix)
melt(mtcars[1:5, 1:3])
describe(mtcars)
skewness(mtcars$mpg)
kurtosis(mtcars$mpg)


set.seed(12310048)
idx <- sample(nrow(df_full), 10000)
df  <- df_full[idx, ]
rownames(df) <- NULL

cat("======================================================================\n")
cat("FLIGHT DATA ANALYSIS — R\n")
cat(sprintf("Full dataset : %d rows × %d columns\n", nrow(df_full), ncol(df_full)))
cat(sprintf("Sample (seed 12310048): %d rows\n", nrow(df)))
cat("======================================================================\n")

str(df)
head(df)


# Q1. Coach Ticket Price Distribution — high, low, average; is $500 good?


cp <- df$coach_price
cat(sprintf("  Min       : $%.2f\n", min(cp)))
cat(sprintf("  Max       : $%.2f\n", max(cp)))
cat(sprintf("  Mean      : $%.2f\n", mean(cp)))
cat(sprintf("  Median    : $%.2f\n", median(cp)))
cat(sprintf("  Std Dev   : $%.2f\n", sd(cp)))
cat(sprintf("  Q1        : $%.2f\n", quantile(cp, 0.25)))
cat(sprintf("  Q3        : $%.2f\n", quantile(cp, 0.75)))
pct_500 <- mean(cp > 500) * 100
cat(sprintf("\n  Flights above $500 : %.1f%%\n", pct_500))
cat(sprintf("\n  Interpretation:\n"))
cat(sprintf("  Mean=%.2f, Median=%.2f. Only %.1f%% cost over $500.\n",
            mean(cp), median(cp), pct_500))
cat("  $500 is well above average — an expensive coach ticket.\n")

# Plots
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

hist(cp, breaks = 40, col = "steelblue", border = "white",
     main = "Q1 – Histogram of Coach Prices",
     xlab = "Coach Price ($)", ylab = "Count")
abline(v = mean(cp),   col = "red",    lty = 2, lwd = 2)
abline(v = median(cp), col = "orange", lty = 2, lwd = 2)
abline(v = 500,        col = "green",  lty = 2, lwd = 2)
legend("topleft",
       legend = c(sprintf("Mean $%.0f", mean(cp)),
                  sprintf("Median $%.0f", median(cp)),
                  "$500 mark"),
       col = c("red", "orange", "green"), lty = 2, cex = 0.8)

boxplot(cp, col = "steelblue", border = "navy",
        main = "Q1 – Boxplot of Coach Prices", ylab = "Coach Price ($)")
abline(h = 500, col = "green", lty = 2, lwd = 2)
legend("topright", legend = "$500 mark", col = "green", lty = 2, cex = 0.8)

par(mfrow = c(1, 1))


# Q2. 8-Hour Flight Prices

df8 <- df[df$hours == 8, "coach_price"]
cat(sprintf("  Number of 8-hr flights : %d\n", length(df8)))
cat(sprintf("  Min    : $%.2f\n", min(df8)))
cat(sprintf("  Max    : $%.2f\n", max(df8)))
cat(sprintf("  Mean   : $%.2f\n", mean(df8)))
cat(sprintf("  Median : $%.2f\n", median(df8)))
pct8_500 <- mean(df8 > 500) * 100
cat(sprintf("  Above $500 : %.1f%%\n", pct8_500))
cat(sprintf("\n  Interpretation:\n"))
cat(sprintf("  Mean for 8-hr flights = $%.2f (vs overall $%.2f).\n", mean(df8), mean(cp)))
cat(sprintf("  %.1f%% cost over $500 — $500 is more reasonable here.\n", pct8_500))

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
hist(df8, breaks = 20, col = "darkorange", border = "white",
     main = "Q2 – 8-Hour Flight Coach Prices", xlab = "Coach Price ($)")
abline(v = mean(df8), col = "red",   lty = 2, lwd = 2)
abline(v = 500,       col = "green", lty = 2, lwd = 2)
legend("topleft", legend = c("Mean", "$500"), col = c("red", "green"), lty = 2, cex = 0.8)

boxplot(df8, col = "darkorange", main = "Q2 – Boxplot: 8-Hour Flights",
        ylab = "Coach Price ($)")
abline(h = 500, col = "green", lty = 2, lwd = 2)
par(mfrow = c(1, 1))


# Q3. Flight Delay Distribution

delay <- df$delay
cat(sprintf("  Mean delay   : %.2f min\n", mean(delay)))
cat(sprintf("  Median delay : %.2f min\n", median(delay)))
cat(sprintf("  Std Dev      : %.2f min\n", sd(delay)))
cat(sprintf("  Max delay    : %.0f min\n",  max(delay)))
cat(sprintf("  No delay (0) : %.1f%%\n",   mean(delay == 0) * 100))
cat(sprintf("  > 15 min     : %.1f%%  (connection risk)\n", mean(delay > 15) * 100))
cat(sprintf("  > 60 min     : %.1f%%  (severe delay)\n",     mean(delay > 60) * 100))
cat("\n  Interpretation:\n")
cat("  Distribution is right-skewed — most delays are short or zero.\n")
cat(sprintf("  %.1f%% of flights risk missing connections.\n", mean(delay > 15) * 100))

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
hist(delay, breaks = 40, col = "tomato", border = "white",
     main = "Q3 – Histogram of Delays", xlab = "Delay (minutes)")
abline(v = 15, col = "orange",  lty = 2, lwd = 2)
abline(v = 60, col = "darkred", lty = 2, lwd = 2)
legend("topright", legend = c("15 min", "60 min"),
       col = c("orange", "darkred"), lty = 2, cex = 0.8)

boxplot(delay, col = "tomato", main = "Q3 – Boxplot of Delays",
        ylab = "Delay (minutes)")
par(mfrow = c(1, 1))


# Q4. Correlation: Coach Price vs Miles, Passengers, Delay, Hours


num_vars <- c("miles", "passengers", "delay", "hours")
for (v in num_vars) {
  ct <- cor.test(df[[v]], df$coach_price)
  sig <- ifelse(ct$p.value < 0.05, "✓ significant", "✗ not significant")
  cat(sprintf("  coach_price ~ %-12s: r = %+.4f  p = %.4f  %s\n",
              v, ct$estimate, ct$p.value, sig))
}
cat("\n  Interpretation:\n")
cat("  Miles and hours show the strongest positive correlations.\n")
cat("  Delay is negligibly and negatively correlated with price.\n")

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
colors_q4 <- c("steelblue", "darkorange", "tomato", "seagreen")
for (i in seq_along(num_vars)) {
  v <- num_vars[i]
  plot(df[[v]], df$coach_price, pch = 16, cex = 0.3, col = adjustcolor(colors_q4[i], 0.4),
       xlab = v, ylab = "Coach Price ($)",
       main = sprintf("Price vs %s (r=%.3f)", v, cor(df[[v]], df$coach_price)))
  abline(lm(df$coach_price ~ df[[v]]), col = "black", lwd = 1.5)
}
par(mfrow = c(1, 1))

# ══════════════════════════════════════════════════════════════════════════════
# Q5. Coach Price vs First-Class Price


ct5 <- cor.test(df$coach_price, df$firstclass_price)
ratio <- mean(df$firstclass_price / df$coach_price)
cat(sprintf("  Pearson r = %.4f,  p-value = %.4e\n", ct5$estimate, ct5$p.value))
cat(sprintf("  Avg first-class / coach ratio : %.2fx\n", ratio))
cat("\n  Interpretation:\n")
cat(sprintf("  Strong positive correlation (r=%.3f). Higher coach always means\n", ct5$estimate))
cat(sprintf("  higher first-class. First-class is ~%.1fx the coach fare.\n", ratio))

plot(df$coach_price, df$firstclass_price,
     pch = 16, cex = 0.3, col = adjustcolor("purple", 0.3),
     xlab = "Coach Price ($)", ylab = "First-Class Price ($)",
     main = sprintf("Q5 – Coach vs First-Class (r=%.3f)", ct5$estimate))
abline(lm(firstclass_price ~ coach_price, data = df), col = "black", lwd = 2)


# Q6. Coach Price vs In-Flight Features


features <- c("inflight_meal", "inflight_entertainment", "inflight_wifi")
for (feat in features) {
  g_yes <- df[df[[feat]] == "Yes", "coach_price"]
  g_no  <- df[df[[feat]] == "No",  "coach_price"]
  tt    <- t.test(g_yes, g_no)
  diff  <- mean(g_yes) - mean(g_no)
  sig   <- ifelse(tt$p.value < 0.05, "✓ sig", "✗ not sig")
  cat(sprintf("  %s\n", feat))
  cat(sprintf("    Yes: $%.2f  |  No: $%.2f  |  Diff: $%+.2f\n",
              mean(g_yes), mean(g_no), diff))
  cat(sprintf("    t = %.3f, p = %.4f  %s\n\n", tt$statistic, tt$p.value, sig))
}
cat("  Interpretation:\n")
cat("  All three raise coach prices significantly. Wifi and entertainment\n")
cat("  have the largest premium (~$68 each), meal adds ~$19.\n")

# Boxplot
boxplot(coach_price ~ inflight_meal + inflight_entertainment + inflight_wifi,
        data = df,
        col = c("lightgreen", "tomato"),
        main = "Q6 – Coach Price by In-Flight Features",
        xlab = "Feature Combination", ylab = "Coach Price ($)",
        las = 2, cex.axis = 0.65)


# Q7. Passengers vs Flight Duration


ct7 <- cor.test(df$hours, df$passengers)
cat(sprintf("  Pearson r = %.4f,  p-value = %.4f\n", ct7$estimate, ct7$p.value))
avg_pass <- tapply(df$passengers, df$hours, mean)
cat("\n  Average passengers by hours:\n")
print(round(avg_pass, 2))
cat("\n  Interpretation:\n")
cat(sprintf("  Very weak correlation (r=%.3f). Passenger counts are relatively\n", ct7$estimate))
cat("  stable across flight durations in this dataset.\n")

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
plot(df$hours, df$passengers,
     pch = 16, cex = 0.3, col = adjustcolor("teal", 0.4),
     xlab = "Hours", ylab = "Passengers",
     main = sprintf("Q7 – Passengers vs Hours (r=%.3f)", ct7$estimate))
abline(lm(passengers ~ hours, data = df), col = "black", lwd = 2)

barplot(avg_pass, col = "teal", border = "white",
        main = "Q7 – Avg Passengers by Hours",
        xlab = "Hours", ylab = "Avg Passengers")
par(mfrow = c(1, 1))


# Q8. Coach & First-Class: Weekend vs Weekday

for (price_col in c("coach_price", "firstclass_price")) {
  wk_yes <- df[df$weekend == "Yes", price_col]
  wk_no  <- df[df$weekend == "No",  price_col]
  tt     <- t.test(wk_yes, wk_no)
  sig    <- ifelse(tt$p.value < 0.05, "✓ sig", "✗ not sig")
  cat(sprintf("  %s:\n    Weekend: $%.2f  |  Weekday: $%.2f  t=%.3f  p=%.4f  %s\n\n",
              price_col, mean(wk_yes), mean(wk_no), tt$statistic, tt$p.value, sig))
}
cat("  Interpretation:\n")
cat("  Weekend flights are significantly more expensive — both coach\n")
cat("  (~$82 premium) and first-class (~$300 premium).\n")

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
boxplot(coach_price ~ weekend, data = df,
        col = c("lightblue", "lightcoral"),
        main = "Q8 – Coach: Weekend vs Weekday",
        xlab = "Weekend", ylab = "Coach Price ($)")

boxplot(firstclass_price ~ weekend, data = df,
        col = c("mediumpurple", "lightyellow"),
        main = "Q8 – First-Class: Weekend vs Weekday",
        xlab = "Weekend", ylab = "First-Class Price ($)")
par(mfrow = c(1, 1))


# Q9. Coach Price: Redeye vs Non-Redeye by Day of Week

day_order <- c("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
pivot <- tapply(df$coach_price, list(df$day_of_week, df$redeye), mean)
pivot <- pivot[day_order, ]
cat("  Average Coach Price by Day × Redeye:\n")
print(round(pivot, 2))
cat("\n  Interpretation:\n")
cat("  Non-redeye flights are consistently pricier. Friday–Sunday\n")
cat("  (weekend effect) show the highest prices regardless of redeye.\n")

# Grouped bar chart
pivot_df <- as.data.frame(pivot)
pivot_df$day <- rownames(pivot_df)
pivot_melt  <- reshape2::melt(pivot_df, id.vars = "day",
                               variable.name = "Redeye", value.name = "avg_price")
pivot_melt$day <- factor(pivot_melt$day, levels = day_order)

p9 <- ggplot(pivot_melt, aes(x = day, y = avg_price, fill = Redeye)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("No" = "steelblue", "Yes" = "tomato")) +
  labs(title = "Q9 – Coach Price: Redeye vs Non-Redeye by Day",
       x = "Day of Week", y = "Avg Coach Price ($)", fill = "Redeye") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
print(p9)


# Q10. COMPREHENSIVE STATISTICAL ANALYSIS

# ── 10a. Summary Statistics ────────────────────────────────────────────────
cat("\n── 10a. Summary Statistics ──\n")

num_cols <- c("miles", "passengers", "delay", "coach_price", "firstclass_price", "hours")
summary_stats <- sapply(df[num_cols], function(x) c(
  Mean   = mean(x),
  Median = median(x),
  SD     = sd(x),
  Min    = min(x),
  Max    = max(x)
))
print(round(summary_stats, 3))

cat("\n  psych::describe() output:\n")
print(describe(df[num_cols]))

# ── 10b. Visualizations ────────────────────────────────────────────────────
cat("\n── 10b. Visualizations ──\n")

# Histograms
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
for (col in num_cols) {
  hist(df[[col]], breaks = 30, col = "steelblue", border = "white",
       main = paste("Histogram –", col), xlab = col, ylab = "Count")
}
par(mfrow = c(1, 1))
mtext("Q10b – Histograms", outer = TRUE, cex = 1.2)

# Boxplots
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
for (col in num_cols) {
  boxplot(df[[col]], col = "steelblue", border = "navy",
          main = paste("Boxplot –", col), ylab = col)
}
par(mfrow = c(1, 1))

# Bar charts — categorical
cat_cols <- c("inflight_meal","inflight_entertainment","inflight_wifi",
              "day_of_week","redeye","weekend")
par(mfrow = c(2, 3), mar = c(5, 4, 3, 1))
for (col in cat_cols) {
  tbl <- sort(table(df[[col]]), decreasing = TRUE)
  barplot(tbl, col = "darkorange", border = "white",
          main = paste("Bar –", col), ylab = "Count",
          las = 2, cex.names = 0.8)
}
par(mfrow = c(1, 1))

# ── 10c. Hypothesis Testing ────────────────────────────────────────────────
cat("\n── 10c. Hypothesis Testing ──\n")

# One-sample t-test: H0 mean coach_price = 380
t_one <- t.test(df$coach_price, mu = 380)
cat(sprintf("\n  One-sample t-test (H0: mean=$380):\n  t=%.4f, p=%.4f  %s\n",
            t_one$statistic, t_one$p.value,
            ifelse(t_one$p.value < 0.05, "Reject H0", "Fail to reject H0")))

# Z-test approximation (large sample — use t-test with df → ∞)
library(BSDA, warn.conflicts = FALSE, quietly = TRUE) # optional; manual if unavailable
z_manual <- (mean(cp) - 350) / (sd(cp) / sqrt(length(cp)))
p_z      <- 2 * pnorm(-abs(z_manual))
cat(sprintf("\n  One-sample Z-test (H0: mean=$350):\n  z=%.4f, p=%.4e  %s\n",
            z_manual, p_z,
            ifelse(p_z < 0.05, "Reject H0", "Fail to reject H0")))

# Chi-square: redeye × inflight_meal
ct_chi <- table(df$redeye, df$inflight_meal)
chi1   <- chisq.test(ct_chi)
cat(sprintf("\n  Chi-square: redeye × inflight_meal\n  χ²=%.4f, df=%d, p=%.4f  %s\n",
            chi1$statistic, chi1$parameter, chi1$p.value,
            ifelse(chi1$p.value < 0.05, "Reject H0 (dependent)", "Fail to reject (independent)")))

# Chi-square: redeye × weekend
ct_chi2 <- table(df$redeye, df$weekend)
chi2r   <- chisq.test(ct_chi2)
cat(sprintf("\n  Chi-square: redeye × weekend\n  χ²=%.4f, df=%d, p=%.4f  %s\n",
            chi2r$statistic, chi2r$parameter, chi2r$p.value,
            ifelse(chi2r$p.value < 0.05, "Reject H0 (dependent)", "Fail to reject (independent)")))

# ── 10d. Independent t-tests ───────────────────────────────────────────────
cat("\n── 10d. Independent t-tests ──\n")

comparisons <- list(
  list(col = "weekend",      label = "Weekend vs Weekday — Coach Price"),
  list(col = "redeye",       label = "Redeye vs Non-Redeye — Coach Price"),
  list(col = "inflight_meal",label = "Meal vs No-Meal — Coach Price")
)
for (comp in comparisons) {
  g1 <- df[df[[comp$col]] == "Yes", "coach_price"]
  g2 <- df[df[[comp$col]] == "No",  "coach_price"]
  tt <- t.test(g1, g2)
  cat(sprintf("\n  %s\n  Yes=$%.2f  No=$%.2f  t=%.3f  p=%.4f  %s\n",
              comp$label, mean(g1), mean(g2), tt$statistic, tt$p.value,
              ifelse(tt$p.value < 0.05, "✓ significant", "✗ not significant")))
}

# ── 10e. Weekend vs Weekday Price Differences ──────────────────────────────
cat("\n── 10e. Price Differences: Weekend vs Weekday ──\n")

for (pc in c("coach_price", "firstclass_price")) {
  wk  <- mean(df[df$weekend == "Yes", pc])
  nwk <- mean(df[df$weekend == "No",  pc])
  cat(sprintf("  %s: Weekend=$%.2f  Weekday=$%.2f  Diff=$%+.2f\n",
              pc, wk, nwk, wk - nwk))
}

# ── 10f. Correlation Matrix ────────────────────────────────────────────────
cat("\n── 10f. Correlation Analysis ──\n")

cor_mat <- cor(df[num_cols], use = "complete.obs", method = "pearson")
print(round(cor_mat, 4))

corrplot(cor_mat,
         method      = "color",
         type        = "upper",
         addCoef.col = "black",
         tl.cex      = 0.85,
         number.cex  = 0.75,
         title       = "Q10f – Correlation Heatmap",
         mar         = c(0, 0, 2, 0))

# ── 10g. Linear Regression — Predict Coach Price ──────────────────────────
cat("\n── 10g. Linear Regression: Predicting Coach Price ──\n")

# Encode binary variables
df_enc <- df
df_enc$inflight_meal          <- ifelse(df$inflight_meal          == "Yes", 1, 0)
df_enc$inflight_entertainment <- ifelse(df$inflight_entertainment == "Yes", 1, 0)
df_enc$inflight_wifi          <- ifelse(df$inflight_wifi          == "Yes", 1, 0)

lm_model <- lm(coach_price ~ miles + passengers + delay + hours +
                  inflight_meal + inflight_entertainment + inflight_wifi,
                data = df_enc)
print(summary(lm_model))

cat(sprintf("\n  Interpretation:\n"))
cat(sprintf("  R²=%.4f — explains %.1f%% of variance in coach_price.\n",
            summary(lm_model)$r.squared,
            summary(lm_model)$r.squared * 100))
cat("  Entertainment and wifi add the most to price (~$68 each).\n")

# Residual plots
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(lm_model, main = "Q10g – OLS Residual Diagnostics")
par(mfrow = c(1, 1))

# ── 10h. Logistic Regression — Predict Redeye ─────────────────────────────
cat("\n── 10h. Logistic Regression: Predicting Redeye Flights ──\n")

df_enc$redeye_bin <- ifelse(df$redeye == "Yes", 1, 0)
logit_model <- glm(redeye_bin ~ coach_price + miles + hours + passengers + delay,
                   data   = df_enc,
                   family = binomial(link = "logit"))
print(summary(logit_model))

# Accuracy
pred_prob  <- predict(logit_model, type = "response")
pred_class <- ifelse(pred_prob >= 0.5, 1, 0)
accuracy   <- mean(pred_class == df_enc$redeye_bin)
cat(sprintf("\n  Classification Accuracy : %.2f%%\n", accuracy * 100))
cat("\n  Interpretation:\n")
cat("  Logit model predicts redeye status. Significant predictors\n")
cat("  (p<0.05) meaningfully shift the log-odds of a redeye flight.\n")


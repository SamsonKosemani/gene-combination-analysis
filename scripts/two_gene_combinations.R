# --------- Two-Gene Combination Analysis ---------
library(dplyr)
library(purrr)
library(readr)
library(ggplot2)
library(combinat)

# Load data
expr_mat <- as.matrix(read.csv("data/expression_data.csv", row.names=1))
phenotype <- read.csv("data/phenotype.csv")$phenotype

# Load thresholds from script 1
single_results <- read_csv("results/single_gene_results.csv")
top_genes <- single_results$gene[1:20]
thresholds <- single_results$best_threshold[match(top_genes, single_results$gene)]

# Function: Test a pair
combine_two_genes <- function(g1, g2, t1, t2, expr_mat, phenotype) {
  high1 <- expr_mat[, g1] >= quantile(expr_mat[, g1], t1)
  high2 <- expr_mat[, g2] >= quantile(expr_mat[, g2], t2)
  group <- high1 & high2
  if (sum(group) == 0 | sum(!group) == 0) return(NULL)
  p <- tryCatch(ks.test(phenotype[group], phenotype[!group])$p.value, error = function(e) NA)
  tibble(gene1=g1, gene2=g2, threshold1=t1, threshold2=t2, p_value=p)
}

# All pairs
pairs <- combn(top_genes, 2, simplify=FALSE)
results <- map_dfr(pairs, function(pair) {
  idx1 <- match(pair[1], top_genes)
  idx2 <- match(pair[2], top_genes)
  combine_two_genes(pair[1], pair[2], thresholds[idx1], thresholds[idx2], expr_mat, phenotype)
})
results <- results %>% filter(!is.na(p_value)) %>% arrange(p_value)
write_csv(results, "results/two_gene_results.csv")
print(head(results, 10))

# Plot for top pair
row <- results[1,]
g1 <- row$gene1
g2 <- row$gene2
t1 <- row$threshold1
t2 <- row$threshold2
high1 <- expr_mat[, g1] >= quantile(expr_mat[, g1], t1)
high2 <- expr_mat[, g2] >= quantile(expr_mat[, g2], t2)
group <- high1 & high2
d <- tibble(value = c(phenotype[group], phenotype[!group]),
            group = rep(c("DoubleHigh", "Other"), c(sum(group), sum(!group))))
p <- ggplot(d, aes(x=value, fill=group)) +
  geom_density(alpha=0.5) +
  labs(title=sprintf("%s+%s (p=%.2e)", g1, g2, row$p_value))
ggsave(sprintf("plots/two_gene/%s_%s_KS.png", g1, g2), p)

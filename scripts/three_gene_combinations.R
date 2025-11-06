# --------- Three-Gene Combination Analysis ---------
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
top_genes <- single_results$gene[1:10]
thresholds <- single_results$best_threshold[match(top_genes, single_results$gene)]

# Function: Test a triplet
combine_three_genes <- function(g1, g2, g3, t1, t2, t3, expr_mat, phenotype) {
  h1 <- expr_mat[, g1] >= quantile(expr_mat[, g1], t1)
  h2 <- expr_mat[, g2] >= quantile(expr_mat[, g2], t2)
  h3 <- expr_mat[, g3] >= quantile(expr_mat[, g3], t3)
  group <- h1 & h2 & h3
  if (sum(group) == 0 | sum(!group) == 0) return(NULL)
  p <- tryCatch(ks.test(phenotype[group], phenotype[!group])$p.value, error = function(e) NA)
  tibble(gene1=g1, gene2=g2, gene3=g3, t1=t1, t2=t2, t3=t3, p_value=p)
}

triplets <- combn(top_genes, 3, simplify=FALSE)
results <- map_dfr(triplets, function(trio) {
  idx <- match(trio, top_genes)
  combine_three_genes(trio[1], trio[2], trio[3], thresholds[idx[1]], thresholds[idx[2]], thresholds[idx[3]], expr_mat, phenotype)
})
results <- results %>% filter(!is.na(p_value)) %>% arrange(p_value)
write_csv(results, "results/three_gene_results.csv")
print(head(results, 10))

# Plot for top triplet
row <- results[1,]
h1 <- expr_mat[, row$gene1] >= quantile(expr_mat[, row$gene1], row$t1)
h2 <- expr_mat[, row$gene2] >= quantile(expr_mat[, row$gene2], row$t2)
h3 <- expr_mat[, row$gene3] >= quantile(expr_mat[, row$gene3], row$t3)
group <- h1 & h2 & h3
d <- tibble(value = c(phenotype[group], phenotype[!group]),
            group = rep(c("TripleHigh", "Other"), c(sum(group), sum(!group))))
p <- ggplot(d, aes(x=value, fill=group)) +
  geom_density(alpha=0.5) +
  labs(title=sprintf("%s+%s+%s (p=%.2e)", row$gene1, row$gene2, row$gene3, row$p_value))
ggsave(sprintf("plots/three_gene/%s_%s_%s_KS.png", row$gene1, row$gene2, row$gene3), p)

#!/usr/bin/env Rscript
# MetaCell-1 (Baran et al. 2019, Genome Biology 20:206) on one pile, run through
# the authors' own engine: tgstat (CRAN; the C++ behind tanaylab/metacell).
#
# The three stages are the MetaCell R package's own functions, copied verbatim
# from tanaylab/metacell with the scdb object store removed -- the package keeps
# every intermediate in an on-disk database keyed by id, which is all the glue
# code does.  The tgstat calls and their arguments are unchanged:
#
#   1. mcell_add_cgraph_from_mat_bknn   (R/cgraph_knn.r + R/utils.r tgs_cor_graph)
#        x_knn = tgs_cor_knn(feat, feat, knn = K * k_expand)
#        gr    = tgs_graph(x_knn, K, k_expand, k_beta)
#   2. mcell_coclust_from_graph_resamp  (R/coclust_graph_cov.r)
#        tgs_graph_cover_resample(edges, knn = round(n_edges / n_cells),
#                                 min_mc_size, cooling, burn_in,
#                                 p_resamp, n_resamp, method = "full")
#   3. mcell_mc_from_coclust_balanced   (R/mc_from_coclust.r)
#        mcell_coclust_filt_by_k_deg(K, alpha) -> tgs_graph_cover(min_mc_size)
#
# Called by step2e_metacell1_driver.py, which builds the input k-NN and saves
# the pile; run that, not this file.
#
# ONE deliberate substitution, in stage 1: x_knn is not tgs_cor_knn on log UMIs
# of feature genes but the exact Euclidean k-NN in the embedding, computed by the
# Python driver and handed over in tgs_cor_knn's own output format (col1, col2,
# cor, rank; self excluded; rank 1 = nearest).  tgs_graph reads only col1, col2
# and rank, so the graph construction downstream is the package's, untouched.
#
# Seeds: every stage calls .set_seed(get_param("mc_rseed")) = set.seed(42), as
# the package does; tgstat derives each resample's seed from R's RNG.
#
# Output (little-endian int32, one per cell, pile-local order): the MetaCell
# cluster id, 0 = outlier (tgs_graph_cover leaves the node uncovered), else 1..M.

suppressPackageStartupMessages(library(tgstat))

a = commandArgs(trailingOnly = TRUE)
kv = do.call(rbind, strsplit(a, "=", fixed = TRUE))
P = setNames(kv[, 2], sub("^--", "", kv[, 1]))
num = function(k) as.numeric(P[[k]])

n          = as.integer(P[["n"]])
K          = as.integer(P[["K"]])
k_expand   = num("k_expand")
k_beta     = num("k_beta")
resamp_min = as.integer(P[["resamp_min_mc_size"]])
p_resamp   = num("p_resamp")
n_resamp   = as.integer(P[["n_resamp"]])
K_bal      = as.integer(P[["balanced_K"]])
alpha      = num("balanced_alpha")
min_mc     = as.integer(P[["min_mc_size"]])
cool       = num("cooling")
burn       = as.integer(P[["burn_in"]])
rseed      = as.integer(P[["rseed"]])
options(tgs_max.processes = as.integer(P[["nproc"]]))

t0 = Sys.time()
lg = function(...) message(sprintf("[R %6.0fs] ", as.numeric(Sys.time() - t0, units = "secs")), ...)

# ---- stage 1: balanced k-NN graph (tgs_graph on the supplied ranks) -----------
cells = paste0("c", seq_len(n))
rd = function(f, what, sz) { con = file(f, "rb"); on.exit(close(con)); readBin(con, what, n = 1e9, size = sz) }
col1 = rd(P[["knn_col1"]], "integer", 4L)
col2 = rd(P[["knn_col2"]], "integer", 4L)
rank = rd(P[["knn_rank"]], "integer", 4L)
dist = rd(P[["knn_dist"]], "double", 4L)
stopifnot(length(col1) == length(col2), length(col1) == length(rank), length(col1) == length(dist),
          min(col1) >= 1, max(col1) <= n, max(col2) <= n, all(col1 != col2))
fac = function(i) structure(i, levels = cells, class = "factor")
x_knn = data.frame(col1 = fac(col1), col2 = fac(col2), cor = -dist, rank = rank)
rm(col1, col2, rank, dist)
lg("x_knn: ", nrow(x_knn), " rows (", n, " cells x K*k_expand = ", K * k_expand, ")")

gr = tgs_graph(x_knn, K, k_expand, k_beta)
rm(x_knn); invisible(gc())
colnames(gr) = c("mc1", "mc2", "w")
lg("balanced graph: ", nrow(gr), " edges, mean out-degree ", round(nrow(gr) / n, 1),
   ", nodes with an edge ", length(unique(c(as.character(gr$mc1), as.character(gr$mc2)))))

# ---- stage 2: mcell_coclust_from_graph_resamp ----------------------------------
# --coc_cache (optional, not in the package): stage 3 parameters can be swept on
# one pile without redoing the resampling, which is ~90% of the runtime.  The
# cache depends only on stages 1-2 inputs, so it is reused only for the same
# pile, K, k_expand, k_beta and resampling arguments.
cache = if ("coc_cache" %in% names(P)) P[["coc_cache"]] else ""
if (nzchar(cache) && file.exists(cache)) {
	coc = readRDS(cache)
	lg("co-cluster loaded from cache ", cache, ": ", nrow(coc$coclust), " pairs")
} else {
	set.seed(seed = rseed)
	edges = gr
	colnames(edges) = c("col1", "col2", "weight")
	K_res = round(nrow(edges) / length(cells))
	resamp = tgs_graph_cover_resample(edges, knn = K_res, resamp_min, cooling = cool, burn_in = burn,
	                                  p_resamp = p_resamp, n_resamp = n_resamp, method = "full")
	resamp$co_cluster = resamp$co_cluster[as.numeric(resamp$co_cluster$node1) > 0, ]
	coc = list(coclust = resamp$co_cluster, n_samp = resamp$samples)
	lg("resampling: ", n_resamp, " x p=", p_resamp, ", knn=", K_res, ", co-cluster pairs ", nrow(coc$coclust))
	if (nzchar(cache)) saveRDS(coc, cache)
}

# ---- stage 3: mcell_coclust_filt_by_k_deg + mcell_mc_from_coclust_balanced ----
mcell_coclust_filt_by_k_deg = function(coc, K, alpha)
{
	edges = coc$coclust

	deg_wgt = as.matrix(table(c(edges$node1, edges$node2), c(edges$cnt,edges$cnt)))
	deg_cum = t(apply(deg_wgt, 1, function(x) cumsum(rev(x))))
	thresh_Kr = rowSums(deg_cum > K)
	lev2int = 1:length(levels(edges$node1))
	thresh_K = rep(NA, length(levels(edges$node1)))
	names(thresh_K) = levels(edges$node1)
	if (is.na(sum(as.numeric(names(thresh_Kr))))){
		thresh_K[names(thresh_Kr)] = thresh_Kr
	}
	else {
		thresh_K[as.numeric(names(thresh_Kr))] = thresh_Kr
	}

	filt_edges = thresh_K[edges$node1] < edges$cnt * alpha |
							thresh_K[edges$node2] < edges$cnt * alpha

	return(filt_edges)
}

set.seed(seed = rseed)
edges = coc$coclust
filt_edges = mcell_coclust_filt_by_k_deg(coc, K_bal, alpha)
# not in the package: an NA here would index a row of NAs into edges.  Every
# node of coclust is in the table that defines thresh_K, so none is expected.
stopifnot(!anyNA(filt_edges))
lg("filtered ", nrow(edges) - sum(filt_edges), " left with ", sum(filt_edges), " based on co-cluster imbalance")
edges = edges[filt_edges, ]

colnames(edges) = c("col1", "col2", "weight")
edges$weight = edges$weight / max(edges$weight)

edges = edges[edges$col1 != edges$col2, ]
redges = edges[, c("col1", "col2", "weight")]
redges$col1 = edges$col2
redges$col2 = edges$col1
edges = rbind(edges, redges)

node_clust = tgs_graph_cover(edges, min_mc, cooling = cool, burn_in = burn)

# ---- write: cluster per cell in pile order, 0 = outlier ------------------------
cl = integer(n)
idx = match(as.character(node_clust$node), cells)
stopifnot(!anyNA(idx))
cl[idx] = node_clust$cluster
f_out = cl == 0
cl[!f_out] = as.integer(as.factor(cl[!f_out]))
lg("final: ", max(cl), " metacells, ", sum(f_out), " outliers of ", n, " cells")
con = file(P[["out"]], "wb"); writeBin(as.integer(cl), con, size = 4L); close(con)
lg("DONE")

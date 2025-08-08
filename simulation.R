install.packages(c("devtools", "BiocManager", "scran", "edgeR", "kableExtra", "mclust"))
BiocManager::install(c("scran", "edgeR", "scater"))
install.packages("SPARSim_0.9.5.tar.gz", repos=NULL, type="source")
install.packages("dplyr")
install.packages("Seurat")
install.packages("patchwork")

library(SPARSim)
library(dplyr)
library(Seurat)
library(patchwork)

pbmc.data <- Read10X(data.dir="./filtered_gene_bc_matrices/hg19/")
pbmc <- CreateSeuratObject(counts=pbmc.data, project="pbmc3k", min.cells=3, min.features=200)
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern="^MT-")
pbmc <- subset(pbmc, subset=nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
pbmc <- NormalizeData(pbmc, normalization.method="LogNormalize", scale.factor=10000)
pbmc <- FindVariableFeatures(pbmc, selection.method="vst", nfeatures=2000)
pbmc <- ScaleData(pbmc)
pbmc <- RunPCA(pbmc)
pbmc <- FindNeighbors(pbmc, dims=1:10)
pbmc <- FindClusters(pbmc, resolution=0.5)

clusters <- pbmc@meta.data$seurat_clusters
conditions <- split(seq_along(clusters), clusters)

new.cluster.ids <- c("Naive CD4 T", "CD14+ Mono", "Memory CD4 T", "B", "CD8 T", "FCGR3A+ Mono",
    "NK", "DC", "Platelet")
names(new.cluster.ids) <- levels(pbmc)
pbmc <- RenameIdents(pbmc, new.cluster.ids)

pbmc$cell_type <- Idents(pbmc)
labeled_pbmc <- data.frame(pbmc$RNA$counts)
colnames(labeled_pbmc) <- make.unique(as.character(pbmc$cell_type))

SPARSim_sim_param_label <- SPARSim_estimate_parameter_from_data(raw_data=as.matrix(labeled_pbmc),
                                                          norm_data=as.matrix(norm_labeled_pbmc),
                                                          conditions=conditions)

label_sim_result <- SPARSim_simulation(dataset_parameter=SPARSim_sim_param_label)

write.csv(labeled_pbmc, file="original_data.csv", row.names=TRUE)
write.csv(label_sim_result$count_matrix, file="cell_annotation.csv", row.names=TRUE)

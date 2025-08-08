library(dplyr);
library(Seurat);
library(HGNChelper);
library(openxlsx);
source('https://raw.githubusercontent.com/kris-nader/sc-type/master/R/sctype_wrapper.R'); 

# Load the data
original_df <- read.csv('original_data.csv', row.names=1);

# Convert to Seurat object
original <- CreateSeuratObject(
    counts=original_df, 
    project='scType_Annotation',
    min.cells=0, 
    min.features=0);

original <- NormalizeData(original, normalization.method='LogNormalize', scale.factor=10000)
original <- FindVariableFeatures(original, selection.method='vst', nfeatures=2000)
original <- ScaleData(original, features=rownames(original))
original <- RunPCA(original, features=VariableFeatures(object=pbmc))
original <- FindNeighbors(original, dims=1:10)
original <- FindClusters(original, resolution=0.5)
original <- RunUMAP(original, dims=1:10)

print(paste('Original matrix dimensions', dim(original_df)[1], 'x', dim(original_df)[2]))
print(paste('Seurat matrix dimensions', dim(original$RNA$counts)[1], 'x', dim(original$RNA$counts)[2]))

# Assign cell types to each cluster
result <- run_sctype(
    original,
    known_tissue_type='Immune system',
    custom_marker_file='https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx',
    name='sctype_classification',
    plot=TRUE);

# Save results
write.csv(result@meta.data, file='original_sctype_classification.csv', row.names=TRUE)

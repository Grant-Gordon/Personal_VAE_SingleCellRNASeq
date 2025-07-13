#pragma once
#include <string_view>
#include <memory>
#include <array>
#include "Layers/LinearLayer.hpp"
#include "Layers/ReLULayer.hpp"
#include "param_init_util.h"

namespace config {
constexpr int              Global__seed = 0;
constexpr bool             Global__using_metadata = false;
constexpr std::string_view Global__scalar = "float";
constexpr int              Training__epochs = 1;
constexpr int              Training__batch_size = 128;
constexpr float            Training__lr = 0.001;
constexpr std::string_view Training__output_dir = "/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/CVAE/JobOutputs";
constexpr std::string_view Data__data_dir = "/mnt/projects/debruinz_project/july2024_census_data/subset";
constexpr std::string_view Data__counts_file_pattern = "human_counts_*.npz";
constexpr std::string_view Data__metadata_file_pattern = "human_metadata_*.pkl";
constexpr std::string_view Data__vocab_builder_out = "/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/CVAE/preprocessed_data/";
constexpr std::string_view Data__species = "human";
constexpr int              Data__chunks_to_preload = 0;
constexpr int              Data__batches_to_prelaod = 0;
constexpr int              Data__num_features = 1000;
constexpr std::string_view ConditionalHeads__metadata_vocab = "/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/CVAE/preprocessed_data/human_vocab_dict.pkl";
constexpr std::string_view ConditionalHeads_MetadataFields_Assay__type = "onehot";
constexpr std::string_view ConditionalHeads_MetadataFields_AssayOntologyTermId__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_CellType__type = "embedding";
constexpr int              ConditionalHeads_MetadataFields_CellType__embedding_dim = 128;
constexpr std::string_view ConditionalHeads_MetadataFields_CellTypeOntologyTermId__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_DatasetId__type = "embedding";
constexpr int              ConditionalHeads_MetadataFields_DatasetId__embedding_dim = 128;
constexpr std::string_view ConditionalHeads_MetadataFields_DevStage__type = "onehot";
constexpr std::string_view ConditionalHeads_MetadataFields_DevelopmentStage__type = "embedding";
constexpr int              ConditionalHeads_MetadataFields_DevelopmentStage__embedding_dim = 128;
constexpr std::string_view ConditionalHeads_MetadataFields_DevelopmentStageOntologyTermId__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_Disease__type = "embedding";
constexpr int              ConditionalHeads_MetadataFields_Disease__embedding_dim = 128;
constexpr std::string_view ConditionalHeads_MetadataFields_DiseaseOntologyTermId__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_DonorId__type = "embedding";
constexpr int              ConditionalHeads_MetadataFields_DonorId__embedding_dim = 128;
constexpr std::string_view ConditionalHeads_MetadataFields_IsPrimaryData__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_NMeasuredVars__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_Nnz__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_ObservationJoinid__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_RawMeanNnz__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_RawSum__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_RawVarianceNnz__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_SelfReportedEthnicity__type = "embedding";
constexpr int              ConditionalHeads_MetadataFields_SelfReportedEthnicity__embedding_dim = 128;
constexpr std::string_view ConditionalHeads_MetadataFields_SelfReportedEthnicityOntologyTermId__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_Sex__type = "onehot";
constexpr std::string_view ConditionalHeads_MetadataFields_SexOntologyTermId__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_SomaJoinid__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_SuspensionType__type = "onehot";
constexpr std::string_view ConditionalHeads_MetadataFields_Tissue__type = "embedding";
constexpr int              ConditionalHeads_MetadataFields_Tissue__embedding_dim = 128;
constexpr std::string_view ConditionalHeads_MetadataFields_TissueGeneral__type = "embedding";
constexpr int              ConditionalHeads_MetadataFields_TissueGeneral__embedding_dim = 128;
constexpr std::string_view ConditionalHeads_MetadataFields_TissueGeneralOntologyTermId__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_TissueOntologyTermId__type = "IGNORE";
constexpr std::string_view ConditionalHeads_MetadataFields_TissueType__type = "IGNORE";
constexpr int              RandomTestArchitecture_DummyCvae__dummy_value = 5;

template <typename Scalar>
constexpr std::array<std::shared_ptr<Layer<Scalar>>, 7> __model = {
    std::make_shared<LinearLayer<Scalar>>(1000, 512, glorot),
    std::make_shared<RELULayer<Scalar>>(),
    std::make_shared<LinearLayer<Scalar>>(512, 128, glorot),
    std::make_shared<RELULayer<Scalar>>(),
    std::make_shared<LinearLayer<Scalar>>(128, 512, glorot),
    std::make_shared<RELULayer<Scalar>>(),
    std::make_shared<LinearLayer<Scalar>>(512, 1000, glorot),
};
template <typename Scalar>
constexpr Adam<Scalar> Training__optimizer = Adam<Scalar>(1, 2, 3);

template <typename Scalar>
constexpr std::array<std::shared_ptr<Layer<Scalar>>, 7> RandomTestArchitecture_DummyCvae__model = {
    std::make_shared<LinearLayer<Scalar>>(1000, 888, glorot),
    std::make_shared<RELULayer<Scalar>>(),
    std::make_shared<LinearLayer<Scalar>>(888, 888, glorot),
    std::make_shared<RELULayer<Scalar>>(),
    std::make_shared<LinearLayer<Scalar>>(888, 888, glorot),
    std::make_shared<RELULayer<Scalar>>(),
    std::make_shared<LinearLayer<Scalar>>(888, 1000, glorot),
};
}
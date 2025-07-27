#pragma once
#include <string>
#include <string_view>

namespace configV {
constexpr std::string_view Global__build_dir = "/home/grant/research/czi/Personal_VAE_SingleCellRNASeq/cpp_CVAE/Builds/MVP_build/";
constexpr std::string_view Global__CMakeLists_home_dir = "/home/grant/research/czi/Personal_VAE_SingleCellRNASeq/cpp_CVAE/";
constexpr std::string_view Global__cpp_executable = "main_cpp";
constexpr int              Global__seed = 0;
constexpr bool             Global__using_metadata = false;
constexpr float            Global__scalar = 0.0;
constexpr int              Training__epochs = 1;
constexpr int              Training__batch_size = 128;
constexpr float            Training__lr = 0.001;
constexpr std::string_view Training__output_dir = "/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/cpp_CVAE/JobOutputs";
constexpr std::string_view Data__data_dir = "/mnt/projects/debruinz_project/july2024_census_data/subset";
constexpr std::string_view Data__counts_file_pattern = "human_counts_*.npz";
constexpr std::string_view Data__metadata_file_pattern = "human_metadata_*.pkl";
constexpr std::string_view Data__vocab_builder_out = "/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/cpp_CVAE/preprocessed_data/";
constexpr std::string_view Data__species = "human";
constexpr int              Data__chunks_to_preload = 0;
constexpr int              Data__batches_to_preload = 0;
constexpr int              Data__num_features = 999;
constexpr std::string_view ConditionalHeads__metadata_vocab = "/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/cpp_CVAE/preprocessed_data/human_vocab_dict.pkl";
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

}
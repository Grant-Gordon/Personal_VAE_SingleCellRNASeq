from trainer import Trainer
import metadata_preprocessor

def main():
    #############################
    #Data
    DATA_DIR="/mnt/projects/debruinz_project/july2024_census_data/subset"
    META_GLOB="human_metadata_?.pkl"
    EXPR_GLOB="human_counts_?.npz" 
    #Preprocessed metadata 
    RERUN_PREPROCESSOR=True
    PREPROCESSOR_DIR="/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/tied-weightsCAE/Preprocessed_metadata"
    META_FIELDS_VOCABS_FILE_NAME="metadata_vocab.json"
    FIELD_SPECS_FILE_NAME="metadata_field_specs.json"
    META_FIELDS_VOCABS_PATH=f"{PREPROCESSOR_DIR}/{META_FIELDS_VOCABS_FILE_NAME}"  # { field_name: { value: idx, ... }, ... }   
    FIELD_SPECS_PATH=f"{PREPROCESSOR_DIR}/{FIELD_SPECS_FILE_NAME}"      #[ FieldSpec(field=..., cardinality=..., using=..., non_null_fraction=...), ... ]
    INCLUDE_FIELDS=[
        "cell_type",
        "disease",
        "development_stage",
        "dev_stage",
        "sex",
        "self_reported_ethnicity",
        "tissue_general",
        "tissue",
        "assay"
    ]
    PREPROCESSOR_ARGS=[
        '--data-dir', f'{DATA_DIR}',
        '--pattern', f'{META_GLOB}',
        '--save-dir', f'{PREPROCESSOR_DIR}',
        '--vocab-json-name', f'{META_FIELDS_VOCABS_FILE_NAME}',
        '--specs-json-name', f'{FIELD_SPECS_FILE_NAME}',
        '--verbose'
        ]
    for f in INCLUDE_FIELDS:
        PREPROCESSOR_ARGS+= ["--include", f]

    #Training
    LEARNING_RATE=0.001
    BATCH_SIZE=128
    NUM_EPOCHS=2
    BATCH_WORKERS=2
    BATCH_PREFETCH_FACTOR=2
    #Model
    LATENT_DIM=128

    

    #############################
    if RERUN_PREPROCESSOR:
        print(f"RERUN_PROCESSOR:{RERUN_PREPROCESSOR}")
        metadata_preprocessor.main(PREPROCESSOR_ARGS)
    else:
        print(f"Metdata preprocessor was not used. \n\t Using preprocessed JSONS at:\n\t - {FIELD_SPECS_PATH}\n\t - {META_FIELDS_VOCABS_PATH}")

    trainer = Trainer(
        data_dir=DATA_DIR,
        expr_glob=EXPR_GLOB,
        meta_glob=META_GLOB,
        field_specs_path=FIELD_SPECS_PATH,
        meta_fields_vocabs_path=META_FIELDS_VOCABS_PATH,
        learning_rate=LEARNING_RATE,
        batch_size = BATCH_SIZE,
        latent_dim = LATENT_DIM,
        batch_workers=BATCH_WORKERS,
        batch_prefetch_factor=BATCH_PREFETCH_FACTOR
    )
    trainer.train(num_epochs=NUM_EPOCHS)



if __name__=="__main__":
    main()
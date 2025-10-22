#local_main.py
from trainer import Trainer
import metadata_preprocessor

def main():
    #############################
    #Data
    DATA_DIR="/home/grant/research/czi/data/july_census_subset_10_chunks"
    META_GLOB="human_metadata_7.pkl"
    EXPR_GLOB="human_counts_7.npz" 
    #Preprocessed metadata 
    RERUN_PREPROCESSOR=False
    PREPROCESSOR_DIR='/home/grant/research/czi/data/preprocessed_10_chunk_subset'
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
    NUM_EPOCHS=1
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
        classifier_latent_dim=LATENT_DIM
    )
    trainer.train(num_epochs=NUM_EPOCHS)

if __name__=="__main__":
    main()
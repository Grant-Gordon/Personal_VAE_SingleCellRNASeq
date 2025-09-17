from trainer import Trainer

def main():
    #############################
    DATA_DIR="/mnt/projects/debruinz_project/july2024_census_data/subset"
    META_GLOB="human_metadata_?.pkl"
    EXPR_GLOB="human_counts_?.npz" #NOTE: Glob uses ? not * as to only test on first 10 chunks
    #Preprocessed metadata 
    META_FIELDS_VOCABS_PATH="./metadata_vocab.json"  # { field_name: { value: idx, ... }, ... }   
    FIELD_SPECS_PATH="./metadata_field_specs.json"         #[ FieldSpec(field=..., cardinality=..., using=..., non_null_fraction=...), ... ]
    #Training
    LEARNING_RATE=0.001
    BATCH_SIZE=128
    NUM_EPOCHS=1
    #Model
    LATENT_DIM=128

    #############################


    trainer = Trainer(
        data_dir=DATA_DIR,
        expr_glob=EXPR_GLOB,
        meta_glob=META_GLOB,
        field_specs_path=FIELD_SPECS_PATH,
        meta_fields_vocabs_path=META_FIELDS_VOCABS_PATH,
        learning_rate=LEARNING_RATE,
        batch_size = BATCH_SIZE,
        latent_dim = LATENT_DIM
    )
    trainer.train(num_epochs=NUM_EPOCHS)



if __name__=="__main__"():
    main()